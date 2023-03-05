// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_HETERPS

#include <algorithm>
#include <deque>

#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_utils.h"

DECLARE_int32(gpups_dedup_pull_push_mode);

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_PSLIB
void AfsWrapper::init(const std::string& fs_name, const std::string& fs_user,
                      const std::string& pass_wd, const std::string& conf) {
  int ret = afs_handler_.init(fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(),
                              conf.c_str());
  if (ret != 0) {
    LOG(ERROR) << "AFS Init Error";
  }
}

int AfsWrapper::remove(const std::string& path) {
  return afs_handler_.remove(path);
}

int AfsWrapper::mkdir(const std::string& path) {
  return afs_handler_.mkdir(path);
}

std::vector<std::string> AfsWrapper::list(const std::string& path) {
  return afs_handler_.list(path);
}

int AfsWrapper::exist(const std::string& path) {
  return afs_handler_.exist(path);
}

int AfsWrapper::upload(const std::string& local_file,
                       const std::string& afs_file) {
  return afs_handler_.upload_file(local_file, afs_file);
}

int AfsWrapper::download(const std::string& local_file,
                         const std::string& afs_file) {
  return afs_handler_.download_file(local_file, afs_file);
}

int AfsWrapper::touchz(const std::string& path) {
  return afs_handler_.touchz(path);
}

std::string AfsWrapper::cat(const std::string& path) {
  return afs_handler_.cat(path);
}

int AfsWrapper::mv(const std::string& old_path, const std::string& dest_path) {
  return afs_handler_.mv(old_path, dest_path);
}
#endif

std::shared_ptr<PSGPUWrapper> PSGPUWrapper::s_instance_ = NULL;
bool PSGPUWrapper::is_initialized_ = false;
std::mutex PSGPUWrapper::ins_mutex;
#ifdef PADDLE_WITH_PSLIB
void PSGPUWrapper::InitAfsApi(const std::string& fs_name,
                              const std::string& fs_user,
                              const std::string& pass_wd,
                              const std::string& conf) {
  int ret = afs_handler_.init(fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(),
                              conf.c_str());
  if (ret != 0) {
    VLOG(0) << "AFS Init Error";
  }
  use_afs_api_ = 1;
}
#endif

void PSGPUWrapper::PreBuildTask(std::shared_ptr<HeterContext> gpu_task) {
  VLOG(3) << "PSGPUWrapper::BuildGPUPSTask begin";
  platform::Timer timeline;
  timeline.Start();
  int device_num = heter_devices_.size();
  // if (!multi_mf_dim_) {
  //  gpu_task->init(thread_keys_shard_num_, device_num);
  //} else {
    gpu_task->init(thread_keys_shard_num_, device_num, multi_mf_dim_);
  //}

  std::vector<std::thread> threads;

  // data should be in input channel
  // if (!multi_mf_dim_) {
  //  thread_keys_.resize(thread_keys_thread_num_);
  //  for (int i = 0; i < thread_keys_thread_num_; i++) {
  //    thread_keys_[i].resize(thread_keys_shard_num_);
  //  }
  //} else {
    thread_dim_keys_.resize(thread_keys_thread_num_);
    for (int i = 0; i < thread_keys_thread_num_; i++) {
      thread_dim_keys_[i].resize(thread_keys_shard_num_);
      for (int j = 0; j < thread_keys_shard_num_; j++) {
        thread_dim_keys_[i][j].resize(multi_mf_dim_);
      }
    }
  //}

  dataset_mutex_.lock();
  Dataset* cur_dataset = dataset_pipe_.front();
  dataset_pipe_.pop();
  dataset_mutex_.unlock();
  std::string data_set_name = std::string(typeid(*cur_dataset).name());

  double* add_time_info = new double[thread_keys_shard_num_ * multi_mf_dim_];
  for (int i = 0; i < thread_keys_shard_num_ * multi_mf_dim_; i++) {
    add_time_info[i] = 0;
  }

  double* dispatch_time_info = new double[16*2];
  static KeyDup dup_ins;
  if (data_set_name.find("SlotRecordDataset") != std::string::npos) {
    platform::Timer timeline_1;
    timeline_1.Start();
    SlotRecordDataset* dataset = dynamic_cast<SlotRecordDataset*>(cur_dataset);
    auto input_channel = dataset->GetInputChannel();
    VLOG(0) << "buildtask::inputslotchannle size: "
            << input_channel->Size();
    const std::deque<SlotRecord>& vec_data = input_channel->GetData();
    size_t total_len = vec_data.size();
    //自适应调节map的bucket大小
    if (!dup_ins.is_init()) {
      size_t feasign_size = 0;
      for (auto iter = vec_data.begin(); iter != vec_data.end() && iter != vec_data.begin() + 500; iter++) {
        feasign_size += (*iter)->slot_uint64_feasigns_.slot_values.size();
      }
      size_t avg_size = feasign_size / 500;
      size_t uniq_size = total_len * avg_size / 40 * 1.3;
      uniq_size = uniq_size / thread_keys_shard_num_ / multi_mf_dim_;
      if (uniq_size < 10000) uniq_size = 10001;
      uniq_size = uniq_size / 10 * 10 + 1;
      dup_ins.init(thread_keys_shard_num_ * multi_mf_dim_, uniq_size, avg_size, total_len);
    } else {
      dup_ins.reset(total_len);
    }
    timeline_1.Pause();
    auto lxch_step_1 = timeline_1.ElapsedSec();
    timeline_1.Start();

    VLOG(0) << "after dup init";

    const uint32_t commit_id = 5000;
    std::atomic<uint64_t> task_doing_num(0);
    const uint64_t max_task_limit = 6000;
    auto& l_dup_ins = dup_ins;

    auto fill_func = [this, &task_doing_num, add_time_info](std::shared_ptr< std::vector<uint64_t> > task, size_t shard_id) -> void {
      platform::Timer timeline_info;
      timeline_info.Start();
      l_dup_ins.batch_add_keys(shard_id, *task);
      task_doing_num.fetch_sub(1);
      timeline_info.Pause();
      add_time_info[shard_id] += timeline_info.ElapsedUS();
    };
    
    auto first_uniq_func = [this, &task_doing_num, &fill_func, &vec_data, dispatch_time_info](int begin_index, int end_index, int dispach_i) -> void {
      platform::Timer timeline_info;
      platform::Timer timeline_info_1;
      double tt_tt = 0;
      timeline_info.Start();
      std::vector< std::shared_ptr< std::vector<uint64_t> > > cache_key;
      cache_key.resize(thread_keys_shard_num_ * multi_mf_dim_);
      for (auto &iter : cache_key) {
        iter = std::shared_ptr< std::vector<uint64_t> >(new std::vector<uint64_t>());
        iter->reserve(commit_id + 200);
      }
      auto iter_start = vec_data.begin() + begin_index;
      auto iter_end = iter_start + (end_index - begin_index);
      int uniq_thread_size = uniq_thread_pool_.size();

      // VLOG(0) << "thread rank:" << dispach_i << " , uniq_thread_size:" << uniq_thread_size << " , begin index:" << begin_index << ", end_index" << end_index
      //        << ", thread_keys_shard_num" << thread_keys_shard_num_ << ", multi_mf_dim_:" << multi_mf_dim_;

      for (auto iter = iter_start; iter != iter_end; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
        const auto& slot_offset = ins->slot_uint64_feasigns_.slot_offsets;
        
        for (size_t slot_idx = 0; slot_idx < slot_offset_vector_.size(); slot_idx++) {
          for (size_t j = slot_offset[slot_offset_vector_[slot_idx]]; j < slot_offset[slot_offset_vector_[slot_idx] + 1]; j++) {
            int dim_id = slot_index_vec_[slot_idx];
            int shard_id = feasign_v[j] % thread_keys_shard_num_;
            int uniq_index = shard_id * multi_mf_dim_ + dim_id;
            cache_key[uniq_index]->emplace_back(feasign_v[j]);
            if (cache_key[uniq_index]->size() >= commit_id) {
              timeline_info_1.Start();
              while(task_doing_num.load() >= max_task_limit) {
                usleep(500);
                continue;
              }
              timeline_info_1.Pause();
              tt_tt += timeline_info_1.ElapsedUS();
              task_doing_num.fetch_add(1);
              uniq_thread_pool_[shard_id % uniq_thread_size]->enqueue(fill_func, cache_key[uniq_index], uniq_index);
              cache_key[uniq_index] = std::shared_ptr< std::vector<uint64_t> >(new std::vector<uint64_t>());
              cache_key[uniq_index]->reserve(commit_id + 200);
            }
          }
        }
      };
      // VLOG(0) << "thread rank:" << dispach_i << " after iter";
      for (size_t i = 0; i < cache_key.size(); i++) {
        if (cache_key[i]->size() == 0) {
          continue;
        }
        timeline_info_1.Start();
        while(task_doing_num.load() >= max_task_limit) {
          usleep(1000);
          continue;
        }
        timeline_info_1.Pause();
        tt_tt += timeline_info_1.ElapsedUS();
        size_t shard_id = i / multi_mf_dim_;
        task_doing_num.fetch_add(1);
        uniq_thread_pool_[shard_id % uniq_thread_size]->enqueue(fill_func, cache_key[i], i);
      }
      timeline_info.Pause();
      dispatch_time_info[dispach_i*2] = timeline_info.ElapsedUS();
      dispatch_time_info[dispach_i*2 + 1] = tt_tt;
    };

    // VLOG(0) << "before fist uniq func, total_len:" << total_len;
    //开始做数据分发
    const int thread_num = 16;
    size_t per_thread_len = total_len / thread_num + 1;
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_num; i++) {
      size_t start_index = i * per_thread_len;
      size_t end_index = std::min(start_index + per_thread_len, total_len);
      if (start_index < end_index) {
        threads.push_back(std::thread(first_uniq_func, start_index, end_index, i));
      }
    }
    for (auto& t : threads) {
      t.join();
    }

    while (task_doing_num.load() != 0) {
      usleep(1000);
    }
    timeline_1.Pause();
    auto lxch_step_2 = timeline_1.ElapsedSec();
    timeline_1.Start();


    VLOG(0) << "after fist uniq func";

    auto func_uniq = [this, &gpu_task] (int shard_id, int dim_id) -> void {
      auto& result_vec = gpu_task->feature_dim_keys_[shard_id][dim_id];
      int dup_shard_id = shard_id * multi_mf_dim_ + dim_id;
      size_t uniq_size = l_dup_ins.get_uniq_key_size(dup_shard_id);
      //加一个0进来，因为去重集合会忽略到0的feasign，就可能导致结果集里面没有，即使多个0，也无所谓
      if (shard_id == 0 && dim_id + 1 == (int)index_dim_vec_.size() && 0) {
        result_vec.resize(uniq_size + 1);
        result_vec[0] = 0;
        l_dup_ins.trans_to_array(dup_shard_id, result_vec.data() + 1);
      } else {
        result_vec.resize(uniq_size);
        l_dup_ins.trans_to_array(dup_shard_id, result_vec.data());
      }
      
      std::sort(result_vec.begin(), result_vec.end());
    };
    std::vector<std::future<void>> task_futures;
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        task_futures.emplace_back(uniq_thread_pool_[i % uniq_thread_pool_.size()]->enqueue(func_uniq, i, j));
      }
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();

    timeline_1.Pause();
    auto lxch_step_3 = timeline_1.ElapsedSec();
    VLOG(0) << "dedup keys time  init: " << lxch_step_1 
            << "  dedup:" << lxch_step_2 
            << "  sort:" << lxch_step_3;

  } else {
    CHECK(false);
  }
  
  timeline.Pause();
  VLOG(0) << "GpuPs task unique cost " << timeline.ElapsedSec() << " seconds.";
  /*
  for (int i = 0; i < 16; i++) {
    VLOG(0) << "lxch dispatch thread_id: " << i
            << "  all_time: " << dispatch_time_info[2*i]
            << "  wait_time: " << dispatch_time_info[2*i+1];
  }
  for (int i = 0; i < thread_keys_shard_num_ * multi_mf_dim_; i++) {
    VLOG(0) << "lxch add shard_id: " << i
            << " all_time: " << add_time_info[i];
  }
  dup_ins.print_detail();
  */
  delete []add_time_info;
  delete []dispatch_time_info;
}

// === all2all ====
/*
void PSGPUWrapper::FilterPull(std::shared_ptr<HeterContext> gpu_task,
                              std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>& value_dim_ptr;
                              const int shard_id,
                              const int dim_id) {
#ifdef PADDLE_WITH_PSLIB
  auto& shard_keys = gpu_task->feature_dim_keys_[shard_id][dim_id];
  // auto& shard_values = gpu_task->value_dim_ptr_[shard_id][dim_id];
  auto& shard_values = value_dim_ptr[shard_id][dim_id];
  size_t dedup_size = 0;
  for (size_t pos = 0; pos < shard_keys.size(); ++pos) {
    auto& key = shard_keys[pos];
    // 如果这个feasign不属于当前机器，跳过
    if (PartitionKeyForRank(key) != rank_id_) {
      continue;
    }
    // 如果全是属于当前机器，那么这个逻辑总是成立
    // 那么就是保持原样
    if (dedup_size == pos) {
      ++dedup_size;
      continue;
    }
    // 如果有一个不是当前机器，那么下一个feasign，dedup_size = pos - 1
    // 
    shard_keys[dedup_size] = shard_keys[pos];
    ++dedup_size;
  }
  // 直接resize???
  // 经过上面的for循环，前面dedup_size个数据都是本机器的。。。
  // 直接把不是本机器的扔掉了???
  shard_keys.resize(dedup_size);
  shard_values.resize(dedup_size);
#endif
}


// 下面两个函数应该功能都在BuildPull里了
// 得看懂然后融进BuildPull，不排除需要拆分函数
void PSGPUWrapper::MergePull(std::shared_ptr<HeterContext> gpu_task) {
  if (!multi_node_) {
    return;
  }
#ifdef PADDLE_WITH_PSLIB
  platform::Timer timeline;
  timeline.Start();
  // barrier
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }
  gloo_wrapper->Barrier();
  timeline.Pause();

  auto barrier_span = timeline.ElapsedSec();


  timeline.Start();
  auto fleet_ptr = paddle::distributed::FleetWrapper::GetInstance();
  std::vector<std::future<void>> task_futures;

  for (int dim_id = 0; dim_id < multi_mf_dim_; ++dim_id) {

    // 看来是其他机器分发过来的feasign
    // 
    auto pass_values = fleet_ptr->worker_ptr_->TakePassSparseReferedValues(
        table_id_, gpu_task->pass_id_, dim_id);

    if (pass_values == nullptr) {
      continue;
    }
    for (int shard_id = 0; shard_id < thread_keys_shard_num_; ++shard_id) {

      auto& merge_values = pass_values->at(shard_id);

      task_futures.emplace_back(pull_thread_pool_[shard_id]->enqueue(
          [this, &gpu_task, &merge_values](int shard_id, int dim_id) {

            auto& shard_keys = gpu_task->feature_dim_keys_[shard_id][dim_id];
            auto& shard_values = gpu_task->value_dim_ptr_[shard_id][dim_id];



            // 合并其他机器发过来的feasign和本地的feasign
            // 
            size_t dedup_size = shard_keys.size();
            size_t merge_num = merge_values.keys.size();
            size_t total = merge_num + dedup_size;
            shard_keys.resize(total);
            shard_values.resize(total);




            size_t dedup_index = dedup_size;
            uint64_t last_key = shard_keys[0];

            size_t i = 0;
            size_t k = 0;

            int num_ranks = node_size_ - 1;

            // 如果是两个机器
            // merge_key应该是排序过的 ???
            if (num_ranks == 1) {
              while (i < dedup_size && k < merge_num) {
                auto& merge_key = merge_values.keys[k];
                auto& key = shard_keys[i];
                if ((key == merge_key) || (last_key == merge_key)) {
                  ++k;
                  continue; // 去重
                }
                if (key < merge_key) {
                  ++i;
                  continue;                                                                          　 
                }

                // key > merge_key ??
                last_key = merge_key;
                shard_keys[dedup_index] = merge_key; // 放到shard_kyes的最后
                shard_values[dedup_index] =
                    CONV2FEATURE_PTR(merge_values.values[k]);
                ++k;
                ++dedup_index;
              }

              uint64_t& key = shard_keys[dedup_size - 1];
              while (k < merge_num) {
                auto& merge_key = merge_values.keys[k];
                if (key == merge_key || last_key == merge_key) {
                  ++k;
                  continue;
                }
                last_key = merge_key;
                shard_keys[dedup_index] = merge_key;
                shard_values[dedup_index] =
                    CONV2FEATURE_PTR(merge_values.values[k]);
                ++k;
                ++dedup_index;
              }



            } else {
              merge_values.offsets.push_back(merge_num);
              CHECK(merge_values.offsets.size() ==
                    static_cast<size_t>(node_size_));
              std::vector<size_t> ranks_pos(num_ranks);
              for (int rank = 0; rank < num_ranks; ++rank) {
                ranks_pos[rank] = merge_values.offsets[rank];
              }
              ssize_t pos = -1;
              int sel_rank = -1;
              uint64_t min_key = last_key;
              while (i < dedup_size && k < merge_num) {
                auto& key = shard_keys[i];
                if (key < min_key) {
                  ++i;
                  continue;
                }
                if (pos == -1) {
                  for (int rank = 0; rank < num_ranks; ++rank) {
                    size_t& max = merge_values.offsets[rank + 1];
                    size_t& off = ranks_pos[rank];
                    while (off < max) {
                      auto& mkey = merge_values.keys[off];
                      if (key == mkey || last_key == mkey || min_key == mkey) {
                        ++k;
                        ++off;
                        continue;
                      }
                      if (pos == -1 || min_key > mkey) {
                        min_key = mkey;
                        pos = off;
                        sel_rank = rank;
                      }
                      break;
                    }
                  }
                  if (pos == -1) {
                    PADDLE_ENFORCE((k == merge_num),
                                   "shardid=%d, k=%d, merge_num=%d",
                                   shard_id,
                                   k,
                                   merge_num);
                    break;
                  }
                  if (key < min_key) {
                    ++i;
                    continue;
                  }
                }
                if (min_key != key) {
                  last_key = merge_values.keys[pos];
                  shard_keys[dedup_index] = last_key;
                  shard_values[dedup_index] =
                      CONV2FEATURE_PTR(merge_values.values[pos]);
                  ++dedup_index;
                }
                pos = -1;
                ++k;
                ++ranks_pos[sel_rank];
              }
              uint64_t& key = shard_keys[dedup_size - 1];
              while (k < merge_num) {
                if (pos == -1) {
                  for (int rank = 0; rank < num_ranks; ++rank) {
                    size_t& max = merge_values.offsets[rank + 1];
                    size_t& off = ranks_pos[rank];
                    while (off < max) {
                      auto& mkey = merge_values.keys[off];
                      if (key == mkey || last_key == mkey || min_key == mkey) {
                        ++k;
                        ++off;
                        continue;
                      }
                      if (pos == -1 || min_key > mkey) {
                        min_key = mkey;
                        pos = off;
                        sel_rank = rank;
                      }
                      break;
                    }
                  }
                  if (pos == -1) {
                    PADDLE_ENFORCE((k == merge_num),
                                   "shardid=%d, k=%d, merge_num=%d",
                                   shard_id,
                                   k,
                                   merge_num);
                    break;
                  }
                }
                last_key = merge_values.keys[pos];
                shard_keys[dedup_index] = last_key;
                shard_values[dedup_index] =
                    CONV2FEATURE_PTR(merge_values.values[pos]);
                ++dedup_index;
                pos = -1;
                ++k;
                ++ranks_pos[sel_rank];
              }
            }
            shard_keys.resize(dedup_index);
            shard_values.resize(dedup_index);
          },
          shard_id,
          dim_id));
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();

  uint64_t total_key = 0;
  for (int shard_id = 0; shard_id < thread_keys_shard_num_; ++shard_id) {
    for (int dim_id = 0; dim_id < multi_mf_dim_; ++dim_id) {
      total_key += gpu_task->feature_dim_keys_[shard_id][dim_id].size();
    }
  }
  timeline.Pause();
  VLOG(0) << "passid=" << gpu_task->pass_id_
          << ", merge pull sparse from CpuPS into GpuPS total keys "
          << total_key << ", cost " << timeline.ElapsedSec()
          << " seconds, barrier span: " << barrier_span;
#endif
}

void PSGPUWrapper::divide_to_device(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  int device_num = heter_devices_.size();
  std::vector<std::future<void>> task_futures;
  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  auto& local_dim_ptr = gpu_task->value_dim_ptr_;

  auto& device_dim_keys = gpu_task->device_dim_keys_;
  auto& device_dim_ptr = gpu_task->device_dim_ptr_;
  auto& device_dim_mutex = gpu_task->dim_mutex_;
  // auto& device_mutex = gpu_task->mutex_;

  if (multi_mf_dim_) {
    for (size_t dev = 0; dev < device_dim_keys.size(); dev++) {
      device_dim_keys[dev].resize(multi_mf_dim_);
      device_dim_ptr[dev].resize(multi_mf_dim_);
    }
  }

  timeline.Start();
  auto build_pull_dynamic_mf_func = [this,
                                     device_num,
                                     &local_dim_keys,
                                     &local_dim_ptr,
                                     &device_dim_keys,
                                     &device_dim_ptr,
                                     &device_dim_mutex](int i, int j) {
    thread_local std::vector<std::vector<uint32_t>> task_pos(device_num);
    auto& h_dim_keys = local_dim_keys[i][j];
    size_t total_keys_len = h_dim_keys.size();
    for (int i = 0; i < device_num; ++i) {
      task_pos[i].reserve((total_keys_len + device_num - 1) / device_num);
      task_pos[i].clear();
    }
    for (size_t k = 0; k < total_keys_len; k++) {
      int shard = h_dim_keys[k] % device_num;
      task_pos[shard].push_back(k);
    }
    auto& h_dim_ptrs = local_dim_ptr[i][j];
    // allocate local keys to devices
    std::vector<int> shuffle_device = shuffle_int_vector(device_num);
    for (auto dev : shuffle_device) {
      device_dim_mutex[dev][j]->lock();
      auto& dev_pos = task_pos[dev];
      size_t len = dev_pos.size();
      auto& d_dim_keys = device_dim_keys[dev][j];
      auto& d_dim_ptr = device_dim_ptr[dev][j];
      size_t cur = d_dim_keys.size();
      size_t total = cur + len;
      d_dim_keys.resize(total);
      d_dim_ptr.resize(total);
      for (size_t k = 0; k < len; ++k) {
        auto& pos = dev_pos[k];
        d_dim_keys[cur + k] = h_dim_keys[pos];
        CHECK(h_dim_ptrs[pos] != 0)
            << "total=" << total_keys_len << ", pos=" << pos << ", k=" << k
            << ", len=" << len;
        d_dim_ptr[cur + k] = h_dim_ptrs[pos];
      }
      device_dim_mutex[dev][j]->unlock();
    }
  };

  if (multi_mf_dim_) {
    task_futures.clear();
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        int tid = (i * multi_mf_dim_ + j) % device_num_;
        task_futures.emplace_back(
            cpu_work_pool_[tid]->enqueue(build_pull_dynamic_mf_func, i, j));
      }
    }
    for (auto& f : task_futures) {
      f.wait();
    }
  }
  timeline.Pause();
  VLOG(1) << "passid=" << gpu_task->pass_id_
          << ", GpuPs prepare for build hbm cost " << timeline.ElapsedSec()
          << " seconds.";
}
*/

// ===== all2all ======

void PSGPUWrapper::BuildPull(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  std::vector<std::future<void>> task_futures;
  int device_num = heter_devices_.size();
  // auto& local_keys = gpu_task->feature_keys_;
  // auto& local_ptr = gpu_task->value_ptr_;

  static std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>> value_dim_ptr;
  value_dim_ptr.resize(thread_keys_shard_num_);
  for (size_t i = 0; i < value_dim_ptr.size(); i++) {
    value_dim_ptr[i].resize(multi_mf_dim_);
    for (size_t j = 0; j < value_dim_ptr[i].size(); j++) {
      value_dim_ptr[i][j].clear();
    }
  }
  

  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  auto& local_dim_ptr = value_dim_ptr;

  // auto& device_keys = gpu_task->device_keys_;
  // auto& device_vals = gpu_task->device_values_;
  auto& device_dim_keys = gpu_task->device_dim_keys_;
  auto& device_dim_ptr = gpu_task->device_dim_ptr_;
  auto& device_dim_mutex = gpu_task->dim_mutex_;

  if (multi_mf_dim_) {
    for (size_t dev = 0; dev < device_dim_keys.size(); dev++) {
      device_dim_keys[dev].resize(multi_mf_dim_);
      device_dim_ptr[dev].resize(multi_mf_dim_);
    }
  }
  // auto& device_mutex = gpu_task->mutex_;

  std::vector<std::thread> threads(thread_keys_shard_num_);
#ifdef PADDLE_WITH_PSLIB
  auto fleet_ptr = FleetWrapper::GetInstance();
#endif
#ifdef PADDLE_WITH_PSCORE
  auto fleet_ptr = paddle::distributed::FleetWrapper::GetInstance();
#endif


#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  // get day_id: day nums from 1970
  struct std::tm b;
  b.tm_year = year_ - 1900;
  b.tm_mon = month_ - 1;
  b.tm_mday = day_;
  b.tm_min = b.tm_hour = b.tm_sec = 0;
  std::time_t seconds_from_1970 = std::mktime(&b);
  int day_id = seconds_from_1970 / 86400;
  fleet_ptr->pslib_ptr_->_worker_ptr->set_day_id(table_id_, day_id);
#endif
  fleet_ptr->pslib_ptr_->_worker_ptr->acquire_table_mutex(0);

  timeline.Start();

/*
  auto ptl_func = [this, &local_keys, &local_ptr, &fleet_ptr, &gpu_task](int i) {
    size_t key_size = local_keys[i].size();
    int32_t status = -1;
#ifdef PADDLE_WITH_PSLIB
    // auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
    //    reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
    //    local_keys[i].data(), key_size);
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(i,
          reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
          local_keys[i].data(), key_size, gpu_task->pass_id_);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
#ifdef PADDLE_WITH_PSCORE
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->worker_ptr_->PullSparsePtr(
          reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
          local_keys[i].data(), key_size);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << local_keys[i].size();
    }
  };
*/

  auto ptl_dynamic_mf_func = [this, &local_dim_keys, &local_dim_ptr,
                              &fleet_ptr, &gpu_task](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    size_t key_size = local_dim_keys[i][j].size();
    int32_t status = -1;
    int32_t cnt = 0;
    local_dim_ptr[i][j].resize(key_size); // shard i , mf_dim:j
    VLOG(0) << "GpuPs shard: " << i << "mf dim: " << index_dim_vec_[j]
            << " key len: " << key_size;
    while (true) {
      // auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(i,
      //     reinterpret_cast<char**>(local_dim_ptr[i][j].data()), this->table_id_,
      //     local_dim_keys[i][j].data(), key_size, gpu_task->pass_id_);

      VLOG(0) << "yxf add pull sparse i: " << i << " mf_dim: " << this->index_dim_vec_[j];
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->add_pull_sparse_task(i,
          reinterpret_cast<char**>(local_dim_ptr[i][j].data()), this->table_id_,
          local_dim_keys[i][j].data(), key_size, gpu_task->pass_id_, this->index_dim_vec_[j]);
     
      bool flag = true;

      tt.wait();

/*
      VLOG(0) << "test local shard i:" << i << ", mf_id:" << j;
      // =======  for debug remote shard ==========
      // if (i < 34) {
          for(size_t u = 0; u < local_dim_ptr[i][j].size(); u++) {
              paddle::ps::DownpourFixedFeatureValue* downpour_val_ptr = local_dim_ptr[i][j][u];
              // auto* cpu_val = downpour_val_ptr->data();
              // VLOG(0) << "[debug]shard:"<< i << ", mf_id:" << j << "before test downpour size meet requirement!!!";
              size_t d_size = downpour_val_ptr->size();
              if (d_size != 8 && d_size != 24 && d_size != 136) {
                VLOG(0) << "[debug]shard:" << i << "mf_id:" << j << "[debug] d_size not meet requirement!!!";
              }
          }
      // }

      // =======  for debug remote shard ==========

      // bool flag = true;

      // tt.wait();
*/

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]" << ", mf_dim:" << this->index_dim_vec_[j] << ", key_size:" << key_size << ", shard_id:" << i;
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }

    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << local_dim_keys[i][j].size();
    }
#endif
  };

  // if (!multi_mf_dim_) {
  //  for (size_t i = 0; i < threads.size(); i++) {
  //    threads[i] = std::thread(ptl_func, i);
  //  }
  //} else {
    threads.resize(thread_keys_shard_num_ * multi_mf_dim_);

    // std::vector<std::future<void>> task_futures;
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i * multi_mf_dim_ + j] = std::thread(ptl_dynamic_mf_func, i, j);
        // task_futures.emplace_back(pull_thread_pool_[i]->enqueue(ptl_dynamic_mf_func, i, j));
      }
    }
    // for (auto& f : task_futures) {
    //   f.wait();
    // }
    // task_futures.clear();
  // }

  // int test_i = 0;
  for (std::thread& t : threads) {
    t.join();
    // test_i += 1;
  }

  fleet_ptr->pslib_ptr_->_worker_ptr->release_table_mutex(0);

  timeline.Pause();
  VLOG(0) << "pull sparse from CpuPS into GpuPS cost " << timeline.ElapsedSec()
          << " seconds.";

   // InitializeGPU已经设置了multi_node_=1
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  // if (gloo_wrapper->Size() > 1) {
  //   VLOG(0) << "yxf::set multi node";
  //   multi_node_ = 1;
  // }

  if (multi_node_) {
    
    if (!gloo_wrapper->IsInitialized()) {
      VLOG(0) << "GLOO is not inited";
      gloo_wrapper->Init();
    }
    gloo_wrapper->Barrier();
    VLOG(0) << "yxf::GlooBarrier";
  } else {
    VLOG(0) << "yxf::set single node";
  }

  timeline.Start();
  // auto& device_task_keys = gpu_task->device_task_keys_;
  // auto& device_task_ptrs = gpu_task->device_task_ptr_;

//   auto build_pull_dynamic_mf_func = [this, device_num, &local_dim_keys,
//                                 &local_dim_ptr, &device_dim_keys,
//                                 &device_dim_ptr,
//                                 &device_dim_mutex,
//                                 &fleet_ptr](int i, int j) {
// #ifdef PADDLE_WITH_PSLIB
//     std::vector<std::vector<FeatureKey>> task_keys(device_num);
//     std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> task_ptrs(
//         device_num);
//     for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
//       int shard = local_dim_keys[i][j][k] % device_num;
//       task_keys[shard].push_back(local_dim_keys[i][j][k]);
//       task_ptrs[shard].push_back(local_dim_ptr[i][j][k]);
//     }
//     if (this->multi_node_) {
//       // std::vector<std::pair<uint64_t, char*>> record_values;
//       auto* record_values_ptr = fleet_ptr->pslib_ptr_->_worker_ptr->get_remote_values(
//         table_id_, this->index_dim_vec_[j], i);
//       if (record_values_ptr == nullptr) {
//         VLOG(0) << "yxf no record shard_id: " << i;
//       } else {
//         auto record_values = *record_values_ptr;
//         for (size_t k = 0; k < record_values.size(); k++) {
          
//           int shard = local_dim_keys[i][j][k] % device_num;
//           task_keys[shard].push_back(record_values[k].first);
//           task_ptrs[shard].push_back((paddle::ps::DownpourFixedFeatureValue*)(record_values[k].second));
//         }
//       }
      
      
//     }
//     // allocate local keys to devices
//     for (int dev = 0; dev < device_num; dev++) {

//         device_dim_mutex[dev][j]->lock();

//         int len = task_keys[dev].size();
//         int cur = device_dim_keys[dev][j].size();
//         device_dim_keys[dev][j].resize(device_dim_keys[dev][j].size() +
//                                          len);
//         device_dim_ptr[dev][j].resize(device_dim_ptr[dev][j].size() + len);
//         for (int k = 0; k < len; ++k) {
//           device_dim_keys[dev][j][cur + k] = task_keys[dev][k];
//           device_dim_ptr[dev][j][cur + k] = task_ptrs[dev][k];
//         }
//         device_dim_mutex[dev][j]->unlock();
//     }
    

// #endif
//   };


  /*
  auto build_pull_dynamic_mf_func = [this, device_num, &local_dim_keys,
                                &local_dim_ptr, &device_dim_keys,
                                &device_dim_ptr,
                                &device_dim_mutex,
                                &fleet_ptr](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    std::vector<robin_hood::unordered_map<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>> task_keys_ptr(device_num);
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
      int shard = local_dim_keys[i][j][k] % device_num;
      task_keys_ptr[shard][local_dim_keys[i][j][k]] = local_dim_ptr[i][j][k];
    }
    if (this->multi_node_) {
      // std::vector<std::pair<uint64_t, char*>> record_values;
      auto* record_values_ptr = fleet_ptr->pslib_ptr_->_worker_ptr->get_remote_values(
        table_id_, this->index_dim_vec_[j], i);
      if (record_values_ptr == nullptr) {
        VLOG(0) << "yxf no record shard_id: " << i;
      } else {
        auto record_values = *record_values_ptr;
        for (size_t k = 0; k < record_values.size(); k++) {
          
          int shard = local_dim_keys[i][j][k] % device_num;
          if (task_keys_ptr[shard].find(record_values[k].first) == task_keys_ptr[shard].end()) {
            task_keys_ptr[shard][record_values[k].first] = (paddle::ps::DownpourFixedFeatureValue*)(record_values[k].second);
          }
          
        }
      }
      
      
    }
    // allocate local keys to devices
    for (int dev = 0; dev < device_num; dev++) {

        device_dim_mutex[dev][j]->lock();

        int len = task_keys_ptr[dev].size();
        int cur = device_dim_keys[dev][j].size();
        device_dim_keys[dev][j].resize(device_dim_keys[dev][j].size() +
                                         len);
        device_dim_ptr[dev][j].resize(device_dim_ptr[dev][j].size() + len);
        int k = 0; 
        for (auto p : task_keys_ptr[dev]) {
          device_dim_keys[dev][j][cur + k] = p.first;
          device_dim_ptr[dev][j][cur + k] = p.second;
          k++;
        }
        device_dim_mutex[dev][j]->unlock();
    }
#endif
  };
  */

  std::vector<std::shared_ptr<std::atomic<uint32_t>> > device_keys_num;
  device_keys_num.resize(device_num * multi_mf_dim_);
  for (size_t i = 0; i < device_keys_num.size(); i++) {
    device_keys_num[i].reset(new std::atomic<uint32_t>());
    device_keys_num[i]->store(0);
  }
  std::atomic<uint32_t> barrir_sum;
  barrir_sum.store(thread_keys_shard_num_ * multi_mf_dim_);



  // 其实就是把分thread_keys_shard_num改成分device_num的优化
  // 用原子操作做的优化

  auto build_pull_dynamic_mf_func = [this, device_num, &local_dim_keys,
                                &local_dim_ptr, &device_dim_keys,
                                &device_dim_ptr,
                                &device_dim_mutex,
                                &device_keys_num,
                                &barrir_sum,
                                &fleet_ptr](int i, int j) {
#ifdef PADDLE_WITH_PSLIB

    // === all2all ====
    // 不是当前机器的shard直接return
    // for hbm oom
    if (PartitionShardForRank(i) != rank_id_) {
      barrir_sum.fetch_sub(1);
      return;
    }
    // ==== all2all ===  

    //先统计大小
    std::vector<uint32_t> local_keys_num;
    local_keys_num.resize(device_num * multi_mf_dim_);
    for (size_t i = 0; i < local_keys_num.size(); i++) {
      local_keys_num[i] = 0;
    }
    // 分device
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
      uint32_t shard = local_dim_keys[i][j][k] % device_num;
      uint32_t shard_idx = shard * multi_mf_dim_ + j;
      local_keys_num[shard_idx]++;
    }
    for (size_t i = 0; i < local_keys_num.size(); i++) {
      device_keys_num[i]->fetch_add(local_keys_num[i]);
    }

    barrir_sum.fetch_sub(1);

    while (barrir_sum.load() != 0) {
      usleep(200);
    }

    auto fill_func = [] (std::mutex* mutex, std::vector<FeatureKey>& keys, 
                          std::vector<paddle::ps::DownpourFixedFeatureValue*>& values, 
                          std::vector<std::pair<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>>& src_kv,
                          uint32_t keys_sum) -> void {
      mutex->lock();
      if (keys.size() == 0) {
        keys.reserve(keys_sum);
        values.reserve(keys_sum);
      }
      int len = src_kv.size();
      for (int i = 0; i < len; i++) {
        keys.push_back(src_kv[i].first);
        values.push_back(src_kv[i].second);
      }
      mutex->unlock();
    };

    // ====================== insert feasigns pulled by remote workers ===============================================
    std::vector<std::vector<std::pair<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>>> task_kv(device_num);

    // std::vector<robin_hood::unordered_map<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>> task_keys_ptr(device_num);
    // for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
    //  int shard = local_dim_keys[i][j][k] % device_num;
    //  task_keys_ptr[shard][local_dim_keys[i][j][k]] = local_dim_ptr[i][j][k];
    //}
    std::unordered_set<uint64_t> local_keys_set(local_dim_keys[i][j].begin(), local_dim_keys[i][j].end());

    // get remote value
    if (this->multi_node_) {
      // std::vector<std::pair<uint64_t, char*>> record_values;
      auto* record_values_ptr = fleet_ptr->pslib_ptr_->_worker_ptr->get_remote_values(
        table_id_, this->index_dim_vec_[j], i);
      if (record_values_ptr == nullptr) {
        VLOG(0) << "yxf no record shard_id: " << i;
      } else {
        auto record_values = *record_values_ptr;
        for (size_t k = 0; k < record_values.size(); k++) {
           int shard = record_values[k].first % device_num;
           if (local_keys_set.find(record_values[k].first) == local_keys_set.end()) {
             task_kv[shard].push_back(std::make_pair(record_values[k].first, (paddle::ps::DownpourFixedFeatureValue*)record_values[k].second));

             if (task_kv[shard].size() > 5000) {
               uint32_t shard_idx = shard * multi_mf_dim_ + j;
               uint32_t shard_sum = device_keys_num[shard_idx]->load();
               fill_func(device_dim_mutex[shard][j], device_dim_keys[shard][j], device_dim_ptr[shard][j],
                  task_kv[shard], shard_sum);
               task_kv[shard].clear();
             }

           }
          // int shard = local_dim_keys[i][j][k] % device_num;
          // if (task_keys_ptr[shard].find(record_values[k].first) == task_keys_ptr[shard].end()) {
          //  task_keys_ptr[shard][record_values[k].first] = (paddle::ps::DownpourFixedFeatureValue*)(record_values[k].second);
          //}
          
        }
      }
    }
     
    // ====================== insert feasigns pulled by remote workers ===============================================
  
    //回填结果
    // std::vector<std::vector<std::pair<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>>> task_kv(device_num);
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {

      uint32_t shard = local_dim_keys[i][j][k] % device_num;

      task_kv[shard].push_back(std::make_pair(local_dim_keys[i][j][k], local_dim_ptr[i][j][k]));

      if (task_kv[shard].size() > 5000) {
        uint32_t shard_idx = shard * multi_mf_dim_ + j;
        uint32_t shard_sum = device_keys_num[shard_idx]->load();
        fill_func(device_dim_mutex[shard][j], device_dim_keys[shard][j], device_dim_ptr[shard][j],
                  task_kv[shard], shard_sum);
        task_kv[shard].clear();
      }
    }

    for (int dev = 0; dev < device_num; dev++) {
      if (task_kv[dev].size() > 0) {
        uint32_t shard_idx = dev * multi_mf_dim_ + j;
        uint32_t shard_sum = device_keys_num[shard_idx]->load();
        fill_func(device_dim_mutex[dev][j], device_dim_keys[dev][j], device_dim_ptr[dev][j],
                  task_kv[dev], shard_sum);
        task_kv[dev].clear();
      }
    }

#endif
  };

/*
  auto build_func = [device_num, &local_keys,
                     &local_ptr, &device_task_keys, &device_task_ptrs](int i) {
    auto& task_keys = device_task_keys[i];
#ifdef PADDLE_WITH_PSLIB
    auto& task_ptrs = device_task_ptrs[i];
#endif

#ifdef PADDLE_WITH_PSCORE
    auto& task_ptrs = device_task_ptrs[i];
#endif

    for (size_t j = 0; j < local_keys[i].size(); j++) {
      int shard = local_keys[i][j] % device_num;
      task_keys[shard].push_back(local_keys[i][j]);
      task_ptrs[shard].push_back(local_ptr[i][j]);
    }
// #ifdef PADDLE_WITH_PSLIB
//     if (record_status) {
//       size_t local_keys_size = local_keys.size();
//       size_t pass_values_size = pass_values.size();
//       for (size_t j = 0; j < pass_values_size; j += local_keys_size) {
//         auto& shard_values = pass_values[j];
//         for (size_t pair_idx = 0; pair_idx < pass_values[j].size();
//              pair_idx++) {
//           auto& cur_pair = shard_values[pair_idx];
//           int shard = cur_pair.first % device_num;
//           task_keys[shard].push_back(cur_pair.first);
//           task_ptrs[shard].push_back(
//               (paddle::ps::DownpourFixedFeatureValue*)cur_pair.second);
//         }
//       }
//     }
// #endif
  };

  if (!multi_mf_dim_) {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(build_func, i));
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();
    VLOG(0) << "GpuPs build hbmps done";
  }
*/

  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      threads[i * multi_mf_dim_ + j] =
          std::thread(build_pull_dynamic_mf_func, i, j);
    }
  }
  for (std::thread& t : threads) {
    t.join();
  }
  
  timeline.Pause();
  if (this->multi_node_) {
    gloo_wrapper->Barrier();
    for (size_t dev_idx = 0; dev_idx < device_dim_keys.size(); dev_idx++) {
      VLOG(0) << "yxf:::devidx: " << dev_idx << " dev keys: " << device_dim_keys[dev_idx][0].size();
    }
  }
  VLOG(0) << "GpuPs prepare for build hbm cost " << timeline.ElapsedSec()
          << " seconds.";
}

void PSGPUWrapper::BuildGPUTask(std::shared_ptr<HeterContext> gpu_task) {
  int device_num = heter_devices_.size();
  platform::Timer timeline;
  timeline.Start();

  std::vector<size_t> feature_keys_count(device_num);
  size_t size_max = 0;

  // if (!multi_mf_dim_) {
  //  for (int i = 0; i < device_num; i++) {
  //    feature_keys_count[i] = gpu_task->device_keys_[i].size();
  //    VLOG(0) << i << " card contains feasign nums: " << feature_keys_count[i];
  //    size_max = std::max(size_max, feature_keys_count[i]);
  //  }
  // } else {
    // 计算每张卡上的feasign数量
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        feature_keys_count[i] += gpu_task->device_dim_ptr_[i][j].size();
        VLOG(0) << i << " card with dynamic mf dim: " << index_dim_vec_[j] << " dim index: " << j << " contains feasign nums: "
              << gpu_task->device_dim_ptr_[i][j].size();
      }
      VLOG(0) << i << " card with dynamic mf contains feasign nums total: "
              << feature_keys_count[i];
      size_max = std::max(size_max, feature_keys_count[i]);
    }

  // }

  if (HeterPs_) {
    delete HeterPs_;
    HeterPs_ = nullptr;
  }

  if (size_max <= 0) {
    VLOG(0) << "Skip build gpu ps cause feasign nums = " << size_max;
    return;
  }

  std::vector<std::thread> threads(device_num);
  auto* accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  HeterPs_ = HeterPsBase::get_instance(size_max, resource_, accessor_type_, optimizer_type_);

  HeterPs_->set_nccl_comm_and_size(inner_comms_, inter_comms_, node_size_, rank_id_);
  HeterPs_->set_trans_inter_comm(trans_inter_comms_);
  HeterPs_->set_thread_keys_shard_num(thread_keys_shard_num_);

  HeterPs_->set_sparse_sgd(optimizer_config_);
  HeterPs_->set_embedx_sgd(optimizer_config_);

  // auto build_func = [this, &gpu_task, &feature_keys_count](int i) {
  //  VLOG(3) << "building table: " << i;
  //  this->HeterPs_->build_ps(i, gpu_task->device_keys_[i].data(),
  //                           gpu_task->device_values_[i].data(),
  //                           feature_keys_count[i], 500000, 2);
  //  if (feature_keys_count[i] > 0) {
  //    HeterPs_->show_one_table(i);
  //  }
  //};

  // multi-thread process
  auto build_dymf_mem_pool = [this, &gpu_task, &accessor_wrapper_ptr](int i, int j) {
    this->HeterPs_->set_multi_mf_dim(multi_mf_dim_, max_mf_dim_);
    int mf_dim = this->index_dim_vec_[j];
    size_t feature_value_size =
        accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
    VLOG(3) << "build dymf mem pool with device:" << i << " dim:" << mf_dim << " feature_value_size:" << feature_value_size;
    auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
    auto& device_dim_ptrs = gpu_task->device_dim_ptr_[i][j];
    size_t len = device_dim_keys.size();
    CHECK(len == device_dim_ptrs.size());
    this->mem_pools_[i * this->multi_mf_dim_ + j] = new MemoryPool(len, feature_value_size);
  };

  auto build_dymf_hbm_pool = [this, &gpu_task, &accessor_wrapper_ptr](int i, int j) {

    auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
    size_t len = device_dim_keys.size();
    int mf_dim = this->index_dim_vec_[j];
    size_t feature_value_size =
        accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
    auto& mem_pool = this->mem_pools_[i * this->multi_mf_dim_ + j];

    VLOG(0) << "build dymf mem pool with device:" << i << " dim:" << mf_dim << ", len:" << len << " feature_value_size:" << feature_value_size;

    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    this->hbm_pools_[i * this->multi_mf_dim_ + j] = new HBMMemoryPool(mem_pool);
    auto& cur_pool = this->hbm_pools_[i * this->multi_mf_dim_ + j];

    // debug gpu memory
    // std::string before_debug_info = "before buildps device:";
    // before_debug_info.append(std::to_string(i));
    // before_debug_info.append(", j:" + std::to_string(j));
    // debug_gpu_memory_info(i, before_debug_info.c_str());
    
    this->HeterPs_->build_ps(i, device_dim_keys.data(),
                             cur_pool->mem(), len, feature_value_size,
                             500000, 2);
    // debug gpu memory
    // std::string after_debug_info = "after buildps device:";
    // after_debug_info.append(std::to_string(i));
    // after_debug_info.append(", j:" + std::to_string(j));
    // debug_gpu_memory_info(i, after_debug_info.c_str());

    if (device_dim_keys.size() > 0) {
      VLOG(3) << "show table: " << i << " table kv size: " << device_dim_keys.size() << "dim: " << mf_dim << " len: " << len;
      HeterPs_->show_one_table(i);
    }
    delete mem_pool;
  };

  int thread_num = 16;
  auto build_dynamic_mf_func = [this, &gpu_task, thread_num, &accessor_wrapper_ptr](int i, int j, int z) {
    int mf_dim = this->index_dim_vec_[j];
    VLOG(3) << "building table: " << i << "with mf dim: " << mf_dim;

    auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
    auto& device_dim_ptrs = gpu_task->device_dim_ptr_[i][j];

    size_t len = device_dim_keys.size();
    CHECK(len == device_dim_ptrs.size());

    auto& mem_pool = this->mem_pools_[i * this->multi_mf_dim_ + j];

    // ============ add for multi-thread ================ 
    int len_per_thread = len / thread_num;
    int remain = len % thread_num;
    int left = -1, right = -1;
    
    int real_len = len_per_thread;
    if (z < remain) real_len++;
    
    if (z < remain) {
      left = z * (len_per_thread + 1);
      right = left + real_len;
    } else {
      left = remain * (len_per_thread + 1) + (z - remain) * len_per_thread;
      right = left + real_len;
    }
    // ============ add for multi-thread ================ 

    for (int k = left; k < right; k++) {
/*
      FeatureValue* val = (FeatureValue*)(mem_pool->mem_address(k));
      float* ptr_val = device_dim_ptrs[k]->data();
      size_t dim = device_dim_ptrs[k]->size();
      val->delta_score = ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::delta_score_index()];
      val->show = ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::show_index()];
      val->clk = ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::click_index()];
      val->slot = int(ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::slot_index()]);
      val->lr = ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::embed_w_index()];
      val->lr_g2sum = ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::embed_g2sum_index()];
      val->cpu_ptr = (uint64_t)(device_dim_ptrs[k]);

      // TODO(xuefeng) set mf_dim while using DownpourCtrDymfAccessor
      // ptr_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::mf_dim_index()] = float(mf_dim);
      val->mf_dim = mf_dim;
      if (dim > 8) {  // CpuPS alreay expand as mf_dim
        val->mf_size = mf_dim + 1;
        for (int x = 0; x < val->mf_dim + 1; x++) {
          val->mf[x] = ptr_val[x + 8];
        }
      } else {
        val->mf_size = 0;
        for (int x = 0; x < val->mf_dim + 1; x++) {
          val->mf[x] = 0;
        }
      }
*/
      float* val = (float*)mem_pool->mem_address(k);
      accessor_wrapper_ptr->BuildFill(val, device_dim_ptrs[k], cpu_accessor_, mf_dim);
    }
  };

    threads.resize(device_num * multi_mf_dim_);
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i + j * device_num] = std::thread(build_dymf_mem_pool, i, j);
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    // threads.clear();
    // // multi-thread process
    // threads.resize(device_num * multi_mf_dim_ * thread_num);
    // for (int i = 0; i < device_num; i++) {
    //   for (int j = 0; j < multi_mf_dim_; j++) {
    //     for (int k = 0; k < thread_num; k++) {
    //       threads[(i + j * device_num) * thread_num + k] = std::thread(build_dynamic_mf_func, i, j, k);
    //     }
    //   }
    // }
    // for (std::thread& t : threads) {
    //   t.join();
    // }
    // threads.clear();
    // tmp test build mfdim 8 first and then 64
    threads.clear();

    /*
    // 这块没必要分开来，后面改回去
    // multi-thread process
    threads.resize(device_num * thread_num);
    for (int i = 0; i < device_num; i++) {
        for (int k = 0; k < thread_num; k++) {
          threads[i * thread_num + k] = std::thread(build_dynamic_mf_func, i, 0, k);
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    for (int i = 0; i < device_num; i++) {
      for (int k = 0; k < thread_num; k++) {
          threads[i * thread_num + k] = std::thread(build_dynamic_mf_func, i, 1, k);
      }
    }
    */

    threads.resize(device_num * thread_num * multi_mf_dim_);

    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        for (int k = 0; k < thread_num; k++) {
          threads[(i + j * device_num) * thread_num + k] = std::thread(build_dynamic_mf_func, i, j, k);
        }
      }
    }

    for (std::thread& t : threads) {
      t.join();
    }

    threads.clear();

    // tmp test build mfdim 8 first and then 64 end
    threads.resize(device_num * multi_mf_dim_);
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i + j * device_num] = std::thread(build_dymf_hbm_pool, i, j);
      }
    }
    /*
    for (std::thread& t : threads) {
      t.join();
    }
    for (int i = 0; i < device_num; i++) {
      threads[i] = std::thread(build_dymf_hbm_pool, i, 1);
    }
    */

    for (std::thread& t : threads) {
      t.join();
    }

    threads.clear();

  timeline.Pause();
  VLOG(0) << "GpuPs build table total costs: " << timeline.ElapsedSec()
          << " s.";
}

void PSGPUWrapper::LoadIntoMemory(bool is_shuffle) {
  // for (size_t i = 0; i < mg_time_0.size(); i++) {
  //  // VLOG(0) << "yxfffff: gpu id: " << i << " push time: " << mg_time_0[i];
  //  mg_time_0[i] = 0.0;
  //}
  platform::Timer timer;
  VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
  timer.Start();

  // debug gpu memory
  // std::string before_debug_mem_info = "before dataset loadinto memory";
  // debug_gpu_memory_info(before_debug_mem_info.c_str());
  dataset_->LoadIntoMemory();
  // debug gpu memory
  // std::string after_debug_mem_info = "after dataset loadinto memory";
  // debug_gpu_memory_info(after_debug_mem_info.c_str());

  timer.Pause();
  VLOG(0) << "LoadIntoMemory cost: " << timer.ElapsedSec() << "s";

  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }
  gloo_wrapper->Barrier();
  VLOG(0) << "yxf::loadintomemory GlooBarrier";

  if (dataset_->GetMemoryDataSize() > 0) {
    // local shuffle
    if (is_shuffle) {
      dataset_->LocalShuffle();
    }
    InitSlotInfo();
    std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
    gpu_task->Reset();
    gpu_task->pass_id_ = (uint16_t)(dataset_->GetPassID());

    dataset_mutex_.lock();
    dataset_pipe_.push(dataset_);
    dataset_mutex_.unlock();

    data_ready_channel_->Put(gpu_task);
  }
  
  VLOG(0) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
}

void PSGPUWrapper::start_build_thread() {
  running_ = true;
  VLOG(3) << "start build CPU ps thread.";
  pre_build_threads_ = std::thread([this] { pre_build_thread(); });
  buildpull_threads_ = std::thread([this] { build_pull_thread(); });
}

void PSGPUWrapper::pre_build_thread() {
  // prebuild: process load_data
  while (running_) {
    std::shared_ptr<HeterContext> gpu_task = nullptr;
    // VLOG (0) << "[debug] before data ready channel";
    if (!data_ready_channel_->Get(gpu_task)) {
      continue;
    }
    VLOG(0) << "thread PreBuildTask start.";
    platform::Timer timer;
    timer.Start();
    // build cpu ps data process
    PreBuildTask(gpu_task);
    timer.Pause();
    VLOG(0) << "thread PreBuildTask end, cost time: " << timer.ElapsedSec()
            << "s";
    buildcpu_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}

void PSGPUWrapper::build_pull_thread() {
  while (running_) {
    std::shared_ptr<HeterContext> gpu_task = nullptr;
    if (!buildcpu_ready_channel_->Get(gpu_task)) {
      continue;
    }
    VLOG(0) << "thread build pull start.";
    platform::Timer timer;
    timer.Start();
    // build cpu ps data process
    BuildPull(gpu_task);
    timer.Pause();
    VLOG(0) << "thread BuildPull end, cost time: " << timer.ElapsedSec() << "s";
    buildpull_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}

void PSGPUWrapper::build_task() {
  // build_task: build_pull + build_gputask
  std::shared_ptr<HeterContext> gpu_task = nullptr;

  // VLOG(0) << "[debug] before gpu free channel";
  
  // train end, gpu free
  if (!gpu_free_channel_->Get(gpu_task)) {
    return;
  }
  // VLOG(0) << "[debug] before build pull channel";
  // ins and pre_build end
  if (!buildpull_ready_channel_->Get(gpu_task)) {
    return;
  }

  VLOG(0) << "BuildPull start.";
  platform::Timer timer;
  timer.Start();
//  BuildPull(gpu_task);

  BuildGPUTask(gpu_task);
  timer.Pause();
  VLOG(0) << " BuildGPUTask end, cost time: " << timer.ElapsedSec()
          << "s";

  current_task_ = gpu_task;
}

void PSGPUWrapper::BeginPass() {
  platform::Timer timer;
  timer.Start();
  if (current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[BeginPass] current task is not ended."));
  }

  build_task();
  timer.Pause();

  if (current_task_ == nullptr) {
    PADDLE_THROW(platform::errors::Fatal(
        "[BeginPass] after build_task, current task is not null."));
  }

  VLOG(0) << "BeginPass end, cost time: " << timer.ElapsedSec() << "s";
}

void PSGPUWrapper::EndPass() {

  if (!current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[EndPass] current task has been ended."));
  }

  platform::Timer timer;
  timer.Start();
  size_t keysize_max = 0;

  // in case of feasign_num = 0, skip dump_to_cpu
  // if (!multi_mf_dim_) {
  //  for (size_t i = 0; i < heter_devices_.size(); i++) {
  //    keysize_max =
  //        std::max(keysize_max, current_task_->device_keys_[i].size());
  //  }
  //} else {
    for (size_t i = 0; i < heter_devices_.size(); i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        keysize_max =
            std::max(keysize_max, current_task_->device_dim_keys_[i][j].size());
      }
    }
  //}

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  int thread_num = 8; 
  auto dump_pool_to_cpu_func = [this, thread_num, &accessor_wrapper_ptr](int i, int j, int z) {

    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(this->resource_->dev_id(i)));

    auto& hbm_pool = this->hbm_pools_[i * this->multi_mf_dim_ + j];
    auto& device_keys = this->current_task_->device_dim_keys_[i][j];
    size_t len = device_keys.size();
    // ============ add for multi-thread ================ 
    int len_per_thread = len / thread_num;
    int remain = len % thread_num;
    int left = -1, right = -1;
    
    int real_len = len_per_thread;
    if (z < remain) real_len++;
    
    if (z < remain) {
      left = z * (len_per_thread + 1);
      right = left + real_len;
    } else {
      left = remain * (len_per_thread + 1) + (z - remain) * len_per_thread;
      right = left + real_len;
    }
    // ============ add for multi-thread ================ 
    int mf_dim = this->index_dim_vec_[j];
    // size_t feature_value_size =
    //    TYPEALIGN(8, sizeof(FeatureValue) + ((mf_dim + 1) * sizeof(float)));
    size_t feature_value_size =
        accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);

    VLOG(3) << "dump pool to cpu table: " << i
            << "thread id:" << z
            << "with mf dim: " << mf_dim
            << " key_len :" << len
            << " real len:" << real_len
            << " feature_value_size:" << feature_value_size;

    char* test_build_values =
        (char*)malloc(feature_value_size * real_len);
    uint64_t offset = left * feature_value_size;

    cudaMemcpy(test_build_values, hbm_pool->mem() + offset,
               feature_value_size * real_len, cudaMemcpyDeviceToHost);
    CHECK(len == hbm_pool->capacity());
    uint64_t unuse_key = std::numeric_limits<uint64_t>::max();

    // VLOG(0) << "sharid:" << i << ", mf_dim:" << j << ", left:" << left << ", right:" << right;

    for (int  x = left; x < right; ++x) {

      if (device_keys[x] == unuse_key) {
        continue;
      }
      size_t local_offset = (x - left) * feature_value_size;

/*
      FeatureValue* gpu_val = (FeatureValue*)(test_build_values + local_offset);
      auto* downpour_value =
        (paddle::ps::DownpourFixedFeatureValue*)(gpu_val->cpu_ptr);
      int downpour_value_size = downpour_value->size();
     
      if (gpu_val->mf_size > 0 && downpour_value_size == 8) {
        downpour_value->resize(gpu_val->mf_size + downpour_value_size);
      }
      float* cpu_val = downpour_value->data();
      // cpu_val[0] = 0;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::delta_score_index()] = gpu_val->delta_score;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::show_index()] = gpu_val->show;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::click_index()] = gpu_val->clk;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::embed_w_index()] = gpu_val->lr;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::embed_g2sum_index()] = gpu_val->lr_g2sum;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::slot_index()] = gpu_val->slot;
      cpu_val[paddle::ps::DownpourCtrDymfAccessor::DownpourCtrDymfFeatureValue::mf_dim_index()] = gpu_val->mf_size;
      if (gpu_val->mf_size > 0) {
        for (int x = 0; x < gpu_val->mf_dim + 1; x++) {
          if (x + 8 >= int(downpour_value->size())) {
            VLOG(0) << "x: " << x << " size: "<< downpour_value_size;
          }
          cpu_val[x + 8] = gpu_val->mf[x];
        }
      }
*/      
      float* gpu_val = (float*)(test_build_values + local_offset);
      accessor_wrapper_ptr->DumpFill(gpu_val, cpu_accessor_, mf_dim);
    }
    free(test_build_values);
  };
  
  if (multi_mf_dim_) {
    VLOG(0) << "dynamic mf dump pool: multi_mf_dim_: " << multi_mf_dim_;
    size_t device_num = heter_devices_.size();
    std::vector<std::thread> threads(device_num * multi_mf_dim_ * thread_num);
    for (size_t i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        for (int k = 0; k < thread_num; k++) {
          threads[(i + j * device_num) * thread_num + k] = std::thread(dump_pool_to_cpu_func, i, j, k);
        }
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    threads.clear();
  }

  if (keysize_max != 0) {
    HeterPs_->end_pass();
  }

  // debug gpu memory
  // std::string before_gpu_mem_info = "before delete hbm pool";
  // debug_gpu_memory_info(before_gpu_mem_info.c_str());
  for (size_t i = 0; i < hbm_pools_.size(); i++) {
    delete hbm_pools_[i];
  }
  // debug gpu memory
  // std::string after_gpu_mem_info = "after delete hbm pool";
  // debug_gpu_memory_info(after_gpu_mem_info.c_str());

  gpu_task_pool_.Push(current_task_);
  current_task_ = nullptr;
  gpu_free_channel_->Put(current_task_);
  timer.Pause();
  // fleet_ptr->pslib_ptr_->_worker_ptr->clear();
  // auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  // gloo_wrapper->Barrier();
  // timer.Pause();
  // VLOG(1) << "EndPass end, cost time: " << timer.ElapsedSec() << "s";
  // VLOG(1) << "yxf::pull: " << time_1;
  // VLOG(1) << "yxf::pull_1: " << time_2;
  // VLOG(1) << "yxf::push: " << time_3;
  // VLOG(1) << "yxf::push_1: " << time_4;
  VLOG(0) << "EndPass end, cost time: " << timer.ElapsedSec() << "s";
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const int hidden_size) {
/*
  VLOG(3) << "Begine Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::Alloc(place, total_length * sizeof(FeatureValue));
  FeatureValue* total_values_gpu = reinterpret_cast<FeatureValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place));

    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf_key = memory::Alloc(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          static_cast<int>(total_length));
    pull_gpups_timer.Pause();
    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values, total_values_gpu, gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(3) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which GPUPS costs: " << pull_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
*/
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const std::vector<int>& slot_dim, // dimension for each slot
                              const int hidden_size) { // check
  VLOG(3) << "Begine Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();

  size_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);

  size_t feature_value_size = 0;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  // feature_value_size = accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);

  feature_value_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

  VLOG(3) << "PullSparse, total_length:" << total_length << " featurevalue size:" << feature_value_size;

  // auto buf = memory::Alloc(place, total_length * feature_value_size);
  // float* total_values_gpu = reinterpret_cast<float*>(buf->ptr());

  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);

    if (FLAGS_gpups_dedup_pull_push_mode > 0) {
      auto& dev = device_caches_[devid_2_index]; // cache key
      int slot_num = static_cast<int>(slot_lengths.size());

      std::vector<int64_t> slot_lengths_lod;
      slot_lengths_lod.reserve(slot_num + 1);
      slot_lengths_lod.push_back(0);

      int64_t total_length = 0;
      for (int i = 0; i < slot_num; ++i) {
        total_length += slot_lengths[i];
        slot_lengths_lod.push_back(total_length);
      }
      dev.total_key_length = total_length;

      VLOG(3) << "[" << device_id << "]Begin copy keys, key_num["
              << total_length << "] dedup mode";
      auto stream = dynamic_cast<phi::GPUContext*>(
                        platform::DeviceContextPool::Instance().Get(place))
                        ->stream();
      // total_len * 3 
      uint64_t* total_keys = dev.keys_tensor.mutable_data<uint64_t>(
          (total_length * 3) * sizeof(uint64_t), place);

      int* gpu_slot_dims = dev.dims_tensor.mutable_data<int>(
          slot_dim.size() * sizeof(int), place);
      uint64_t** gpu_keys = dev.keys_ptr_tensor.mutable_data<uint64_t*>(
          keys.size() * sizeof(uint64_t*), place);

      int64_t* slot_lens = dev.slot_lens.mutable_data<int64_t>(
          (slot_num + 1) * sizeof(int64_t), place);

      cudaMemcpyAsync(gpu_keys,
                      keys.data(),
                      keys.size() * sizeof(uint64_t*),
                      cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(slot_lens,
                      slot_lengths_lod.data(),
                      slot_lengths_lod.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      stream);

      cudaMemcpyAsync(gpu_slot_dims,
                      slot_dim.data(),
                      slot_dim.size() * sizeof(int),
                      cudaMemcpyHostToDevice,
                      stream);


      // 本来是CopyForPull里的临时开辟的显存，现在拿出来
      // 被持有,是有什么用吗
      float** gpu_values = dev.values_ptr_tensor.mutable_data<float*>(
          values.size() * sizeof(float*), place);
      cudaMemcpyAsync(gpu_values,
                      values.data(),
                      values.size() * sizeof(float*),
                      cudaMemcpyHostToDevice,
                      stream);
      // total_len * 5
      int* key2slot = dev.keys2slot.mutable_data<int>(
          (total_length * 5) * sizeof(int), place);

      // key2slot前面total_length个数保存每个key属于哪个slot
      // total_keys前面total_length个数保存当前pass用的所有key
      this->CopyKeys2(place,
                     gpu_keys,
                     total_keys,
                     slot_lens,
                     slot_num,
                     static_cast<int>(total_length),
                     key2slot);
     
      // ======= pull dedup =========
      uint32_t* d_restore_idx =
          reinterpret_cast<uint32_t*>(&key2slot[total_length]);
      uint32_t* d_sorted_idx =
          reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
      uint32_t* d_offset =
          reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
      uint32_t* d_merged_cnts =
          reinterpret_cast<uint32_t*>(&d_offset[total_length]);

      uint64_t* d_merged_keys =
          reinterpret_cast<uint64_t*>(&total_keys[total_length]);
      uint64_t* d_sorted_keys =
          reinterpret_cast<uint64_t*>(&d_merged_keys[total_length]);

      int dedup_size = HeterPs_->dedup_keys_and_fillidx(
          devid_2_index,
          static_cast<int>(total_length),
          total_keys,     // input
          d_merged_keys,  // output
          d_sorted_keys,  // sort keys
          d_restore_idx,  // pull fill idx
          d_sorted_idx,   // sort old idx
          d_offset,       // offset
          d_merged_cnts,
          FLAGS_gpups_dedup_pull_push_mode & 0x02);

      PADDLE_ENFORCE_GT(dedup_size,
                        0,
                        platform::errors::PreconditionNotMet(
                            "dedup keys need more than zero failed in BoxPS."));
      dev.dedup_key_length = dedup_size;
      // ======= pull dedup =========
      // d_restore_idx表示的key在排序去重以后的idx
      // feature_value_size 就是pull_value_size
      int64_t total_bytes = dedup_size * feature_value_size;
      // FeatureValue* total_values_gpu =
      //    dev.pull_push_tensor.mutable_data<FeatureValue>(total_bytes, place);
      float* total_values_gpu =
          dev.pull_push_tensor.mutable_data<float>(total_bytes, place);
      
      pull_gpups_timer.Start();

      // d_merged_keys和total_values_gpu一一对应
      // total_values_gpu里的每个向量现在是pull_value
      HeterPs_->pull_sparse(
          devid_2_index, d_merged_keys, total_values_gpu, dedup_size);

      auto buf_dim =
          memory::Alloc(place, slot_dim.size() * sizeof(int));

      int* gpu_dim = reinterpret_cast<int*>(buf_dim->ptr());
      cudaMemcpy(gpu_dim, slot_dim.data(),
                slot_dim.size() * sizeof(int), cudaMemcpyHostToDevice);

      // CopyForPull(place,
      //            total_keys,
      //            gpu_values,
      //            total_values_gpu,
      //            slot_lens,
      //            key2slot,
      //            max_mf_dim_ + 3,
      //            total_length,
      //            gpu_slot_dims,
      //            d_restore_idx,
      //            gpu_dim);

      accessor_wrapper_ptr->CopyForPull(place,
                                        total_keys,
                                        gpu_values,
                                        total_values_gpu,
                                        slot_lens,
                                        key2slot,
                                        max_mf_dim_ + 3, // pull_value dim size
                                        total_length,
                                        gpu_slot_dims,
                                        d_restore_idx,
                                        gpu_dim,
                                        feature_value_size);

      // VLOG(0) << "yxffff dedup mode dedup size: " << dedup_size << " origin size: " <<  total_length; 

    } else {

    auto buf = memory::Alloc(place, total_length * feature_value_size);
    float* total_values_gpu = reinterpret_cast<float*>(buf->ptr());
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({int64_t(total_length), 1}, place));
    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf_key = memory::Alloc(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    auto buf_dim =
        memory::Alloc(place, slot_dim.size() * sizeof(int));
    int* gpu_dim = reinterpret_cast<int*>(buf_dim->ptr());
    cudaMemcpy(gpu_dim, slot_dim.data(),
               slot_dim.size() * sizeof(int), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));

    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          total_length);
    
    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";

    // 把total_values_gpu中的数据拷贝到values
    // total_values_gpu会回收
    accessor_wrapper_ptr->CopyForPull(place,
                                      gpu_keys,
                                      values,
                                      total_values_gpu,
                                      gpu_len,
                                      static_cast<int>(slot_lengths.size()),
                                      hidden_size,
                                      total_length,
                                      gpu_dim,
                                      feature_value_size);


    pull_gpups_timer.Pause();
    }
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  // time_1 += all_timer.ElapsedSec();
  // time_2 += pull_gpups_timer.ElapsedSec();
  VLOG(3) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
            << " s, of which pullsparse costs: " << pull_gpups_timer.ElapsedSec()
            << " s";
  VLOG(3) << "End PullSparse";

}

void PSGPUWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                  const int table_id,
                                  const std::vector<const uint64_t*>& keys,
                                  const std::vector<const float*>& grad_values,
                                  const std::vector<int64_t>& slot_lengths,
                                  const int hidden_size, const int batch_size) {
  VLOG(3) << "Begin GPUPS PushSparseGrad";
  platform::Timer all_timer;
  platform::Timer push_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
/*
  VLOG(0) << "yxf pushhhh gpu: " << place.GetDeviceId() << " len: " << total_length;
  size_t grad_value_size =
      TYPEALIGN(8, sizeof(FeaturePushValue) + (max_mf_dim_ * sizeof(float)));
  auto buf = memory::Alloc(place, total_length * grad_value_size);
  VLOG(3) << "Push Sparse Max mf dimention: " << max_mf_dim_;
  FeaturePushValue* total_grad_values_gpu =
      reinterpret_cast<FeaturePushValue*>(buf->ptr());
  int device_id = 0;
*/
  // VLOG(0) << "yxf pushhhh gpu: " << place.GetDeviceId() << " len: " << total_length;

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  // auto buf = memory::Alloc(place, total_length * grad_value_size);
  // VLOG(3) << "Push Sparse Max mf dimention: " << max_mf_dim_ << " grad_value_size: " << grad_value_size;
  // float* total_grad_values_gpu = reinterpret_cast<float*>(buf->ptr());

  int device_id = 0;

  if (platform::is_cpu_place(place)) {

    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));

  } else if (platform::is_gpu_place(place)) {
    device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);



    if (FLAGS_gpups_dedup_pull_push_mode > 0) {
      auto& dev = device_caches_[devid_2_index];
      // 覆盖了
      int64_t total_length = dev.total_key_length;
      VLOG(3) << "Begin push sparse, key_num[" << total_length
              << "] dedup mode, device:" << device_id << ", index"
              << devid_2_index;
      auto stream = dynamic_cast<phi::GPUContext*>(
                        platform::DeviceContextPool::Instance().Get(place))
                        ->stream();

      uint64_t* total_keys = dev.keys_tensor.data<uint64_t>();
      int* slot_dims = dev.dims_tensor.data<int>();

      int slot_num = static_cast<int>(slot_lengths.size());

      if (!dev.d_slot_vector.IsInitialized()) {
        int* buf_slot_vector =
            dev.d_slot_vector.mutable_data<int>(slot_num * sizeof(int), place);
        cudaMemcpyAsync(buf_slot_vector,
                        slot_vector_.data(),
                        slot_num * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream);
      }

      const int64_t* slot_lens = dev.slot_lens.data<int64_t>();
      const int* d_slot_vector = dev.d_slot_vector.data<int>();

      const int* key2slot = dev.keys2slot.data<int>();
      float** gpu_values = dev.values_ptr_tensor.data<float*>();

      cudaMemcpyAsync(gpu_values,
                      grad_values.data(),
                      grad_values.size() * sizeof(float*),
                      cudaMemcpyHostToDevice,
                      stream);
     
      // 保存的是merge以后的keys
      uint64_t* d_merged_keys = &total_keys[total_length];

      int64_t dedup_size = dev.dedup_key_length;
      int64_t total_bytes = dedup_size * grad_value_size;

      float* total_grad_values_gpu =
          dev.pull_push_tensor.mutable_data<float>(total_bytes, place);

      // FeaturePushValue* total_grad_values_gpu =
      //    dev.pull_push_tensor.mutable_data<FeaturePushValue>(total_bytes, place);

      VLOG(1) << "push dedup mode dedup size: " << dedup_size << " origin size: " <<  total_length; 
    
      // 大于3倍的重复率
      // 我感觉应该是才走这个逻辑才对呀??,
      if (total_length > dedup_size * 3) {
        const uint32_t* d_restore_idx =
            reinterpret_cast<const uint32_t*>(&key2slot[total_length]);
        accessor_wrapper_ptr->CopyForPush(place,
                                          total_keys,
                                          gpu_values,
                                          total_grad_values_gpu,
                                          d_slot_vector,
                                          slot_lens,
                                          max_mf_dim_ + 3,
                                          total_length,
                                          dedup_size,
                                          batch_size,
                                          slot_dims,
                                          key2slot,
                                          d_restore_idx,
                                          grad_value_size);
      } else {
        // 保存每个feasign原来的idx
        const uint32_t* d_sorted_idx =
            reinterpret_cast<const uint32_t*>(&key2slot[total_length * 2]);
        // 前缀和
        const uint32_t* d_offset =
            reinterpret_cast<const uint32_t*>(&d_sorted_idx[total_length]);
        // 每个feasign的数量
        const uint32_t* d_merged_cnts =
            reinterpret_cast<const uint32_t*>(&d_offset[total_length]);
        accessor_wrapper_ptr->CopyForPush(place,
                                          d_merged_keys,
                                          gpu_values,
                                          total_grad_values_gpu,
                                          d_slot_vector,
                                          slot_lens,
                                          max_mf_dim_ + 3,
                                          total_length,
                                          dedup_size,
                                          batch_size,
                                          slot_dims,
                                          key2slot,
                                          d_sorted_idx,
                                          d_offset,
                                          d_merged_cnts,
                                          grad_value_size);
      }
      VLOG(1) << "dedup mode dedup size: " << dedup_size << " origin size: " <<  total_length; 
      push_gpups_timer.Start();
      HeterPs_->push_sparse(devid_2_index,
                            d_merged_keys,
                            total_grad_values_gpu,
                            static_cast<int>(dedup_size));
      push_gpups_timer.Pause();
    } else {
      auto buf = memory::Alloc(place, total_length * grad_value_size);
      VLOG(3) << "Push Sparse Max mf dimention: " << max_mf_dim_ << " grad_value_size: " << grad_value_size;
      float* total_grad_values_gpu = reinterpret_cast<float*>(buf->ptr());
      LoDTensor& cached_total_keys_tensor = keys_tensor[devid_2_index];
      uint64_t* total_keys =
          reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());

      VLOG(3) << "Begin copy grad tensor to gpups struct";
      accessor_wrapper_ptr->CopyForPush(place,
                                        grad_values,
                                        total_grad_values_gpu,
                                        slot_lengths,
                                        total_length,
                                        batch_size,
                                        grad_value_size,
                                        slot_vector_,
                                        slot_mf_dim_vector_);

      VLOG(3) << "Begin call PushSparseGPU in GPUPS, dev: " << devid_2_index
              << " len: " << total_length;

      push_gpups_timer.Start();

      HeterPs_->push_sparse(devid_2_index, total_keys, total_grad_values_gpu,
                            static_cast<int>(total_length));

      push_gpups_timer.Pause();
    }

  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  // auto all_time = all_timer.ElapsedSec();
  // mg_time_0[device_id] += all_time;
  // time_3 += all_time;
  // time_4 += push_gpups_timer.ElapsedSec();
  VLOG(3) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which GPUPS cost: " << push_gpups_timer.ElapsedSec()
          << " s" << " gpu_num: " << device_id << " len: " << total_length;
  
  VLOG(3) << "End PushSparseGrad";
}


#ifdef PADDLE_WITH_PSLIB
void add_sparse_optimizer(
    std::unordered_map<std::string, float>& config,  // NOLINT
    const ::paddle::SparseCommonSGDRuleParameter& sgd_param,
    const std::string& prefix = "") {
  auto optimizer_name = sgd_param.name();
  if (optimizer_name == "naive") {
    config[prefix + "optimizer_type"] = 0;
    config[prefix + "learning_rate"] = sgd_param.naive().learning_rate();
    config[prefix + "initial_range"] = sgd_param.naive().initial_range();
    if (sgd_param.naive().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.naive().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.naive().weight_bounds()[1];
    }
  } else if (optimizer_name == "adagrad") {
    config[prefix + "optimizer_type"] = 1;
    config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
    config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
    if (sgd_param.adagrad().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  } else if (optimizer_name == "std_adagrad") {
    config[prefix + "optimizer_type"] = 2;
    config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
    config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
    if (sgd_param.adagrad().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  } else if (optimizer_name == "adam") {
    config[prefix + "optimizer_type"] = 3;
    config[prefix + "learning_rate"] = sgd_param.adam().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adam().initial_range();
    if (sgd_param.adam().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adam().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adam().weight_bounds()[1];
    }
  }
}

void PSGPUWrapper::InitializeGPUServer(const std::string& fleet_desc) {
  // optimizer config for hbmps
  google::protobuf::TextFormat::ParseFromString(fleet_desc, &_ps_param);
  auto sparse_table =
      _ps_param.server_param().downpour_server_param().downpour_table_param(0);
  auto sparse_table_accessor = sparse_table.accessor();
  auto sparse_table_accessor_parameter =
      sparse_table_accessor.downpour_accessor_param();
  auto accessor_class = sparse_table_accessor.accessor_class();

  // NOTE(zhangminxu): gpups' sparse table optimizer config,
  // now only support single sparse table
  // auto sparse_table = param_.sparse_table(0);
  std::unordered_map<std::string, float> config;
  if (accessor_class == "DownpourFeatureValueAccessor" ||
      accessor_class == "DownpourCtrAccessor" ||
      accessor_class == "DownpourCtrDoubleAccessor") {

    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["learning_rate"] =
        sparse_table_accessor.sparse_sgd_param().learning_rate();
    config["initial_g2sum"] =
        sparse_table_accessor.sparse_sgd_param().initial_g2sum();
    config["initial_range"] =
        sparse_table_accessor.sparse_sgd_param().initial_range();
    if (sparse_table_accessor.sparse_sgd_param().weight_bounds_size() == 2) {
      config["min_bound"] =
          sparse_table_accessor.sparse_sgd_param().weight_bounds()[0];
      config["max_bound"] =
          sparse_table_accessor.sparse_sgd_param().weight_bounds()[1];

    }
    // NOTE(zhangminxu): for DownpourCtrAccessor & DownpourCtrDoubleAccessor,
    // optimizer config for embed_w & embedx_w is the same
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold(); // default = 10
    config["mf_learning_rate"] = config["learning_rate"];
    config["mf_initial_g2sum"] = config["initial_g2sum"];
    config["mf_initial_range"] = config["initial_range"];
    config["mf_min_bound"] = config["min_bound"];
    config["mf_max_bound"] = config["max_bound"];
    config["mf_embedx_dim"] = sparse_table_accessor.embedx_dim(); // default = 8

  } else if (accessor_class == "DownpourSparseValueAccessor") {
    auto optimizer_name = sparse_table_accessor.sparse_commonsgd_param().name();
    if (optimizer_name == "naive") {
      config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .learning_rate();
      config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .initial_range();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .naive()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .naive()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .naive()
                                  .weight_bounds()[1];
      }
    } else if (optimizer_name == "adagrad") {
      config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .learning_rate();
      config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .initial_range();
      config["initial_g2sum"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .initial_g2sum();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .adagrad()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adagrad()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adagrad()
                                  .weight_bounds()[1];
      }
    } else if (optimizer_name == "adam") {
      config["learning_rate"] =
          sparse_table_accessor.sparse_commonsgd_param().adam().learning_rate();
      config["initial_range"] =
          sparse_table_accessor.sparse_commonsgd_param().adam().initial_range();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .adam()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adam()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adam()
                                  .weight_bounds()[1];
      }
    }
  } else if (accessor_class == "DownpourUnitAccessor" ||
             accessor_class == "DownpourDoubleUnitAccessor" ||
             accessor_class == "DownpourCtrDymfAccessor") {
    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold();
    // optimizer config for embed_w and embedx
    add_sparse_optimizer(config, sparse_table_accessor.embed_sgd_param());
    add_sparse_optimizer(
        config, sparse_table_accessor.embedx_sgd_param(), "mf_");
    config["mf_embedx_dim"] = sparse_table_accessor.embedx_dim(); // default = 8
  }
  config["sparse_shard_num"] = sparse_table.shard_num();

  GlobalAccessorFactory::GetInstance().Init(accessor_class);

  GlobalAccessorFactory::GetInstance().GetAccessorWrapper()->Configure(
        config);
  
  InitializeGPUServer(config);
  SetCPUAccessorType(accessor_class);
}
#endif

}  // end namespace framework
}  // end namespace paddle
#endif
