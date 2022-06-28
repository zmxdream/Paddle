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
  gpu_task->init(thread_keys_shard_num_, device_num, multi_mf_dim_);
  
  //step1 将读取到的ins，key先做一个粗去重(主要是多线程粗去重)
  thread_keys_.resize(thread_keys_thread_num_);
  for (auto& iter : thread_keys_) {
    iter.resize(thread_keys_thread_num_);
    for (auto& iter1 : iter) {
      iter1.resize(multi_mf_dim_);
      for (auto& iter2 : iter1) {
        iter2.clear();
      }
    }
  }
  dataset_mutex_.lock();
  Dataset* cur_dataset = dataset_pipe_.front();
  dataset_pipe_.pop();
  dataset_mutex_.unlock();

  std::function<void(void*, int, int, int)> first_uniq_func;
  void* record_vec = nullptr;
  size_t total_len = 0;
  std::string data_set_name = std::string(typeid(*cur_dataset).name());
  if (data_set_name.find("SlotRecordDataset") != std::string::npos) {
    SlotRecordDataset* dataset = dynamic_cast<SlotRecordDataset*>(cur_dataset);
    auto input_channel = dataset->GetInputChannel();
    const std::deque<SlotRecord>* vec_data = &(input_channel->GetData());
    record_vec = (void*)(vec_data);
    total_len = vec_data->size();

    first_uniq_func = [this](void* ptr, int begin_index, int end_index, int i) -> void {
      const std::deque<SlotRecord>& total_data = *((const std::deque<SlotRecord>*)ptr);
      for (auto iter = total_data.begin() + begin_index; iter != total_data.begin() + end_index; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
        const auto& slot_offset = ins->slot_uint64_feasigns_.slot_offsets;
        for (size_t slot_idx = 0; slot_idx < slot_offset_vector_.size(); slot_idx++) {
          for (size_t j = slot_offset[slot_offset_vector_[slot_idx]]; j < slot_offset[slot_offset_vector_[slot_idx] + 1]; j++) {
            int shard_id = feasign_v[j] % thread_keys_shard_num_;
            int dim_id = slot_index_vec_[slot_idx];
            if (feasign_v[j] != 0) {
              this->thread_keys_[i][shard_id][dim_id].insert(feasign_v[j]);
            }
          }
        }
      }
    };
  } else {
    CHECK(data_set_name.find("MultiSlotDataset") != std::string::npos);
    MultiSlotDataset* dataset = dynamic_cast<MultiSlotDataset*>(cur_dataset);
    auto input_channel = dataset->GetInputChannel();
    const std::deque<Record>* vec_data = &(input_channel->GetData());
    record_vec = (void*)(vec_data);
    total_len = vec_data->size();
    first_uniq_func = [this](void* ptr, int begin_index, int end_index, int i) -> void {
      const std::deque<Record>& total_data = *((const std::deque<Record>*)ptr);
      for (auto iter = total_data.begin() + begin_index; iter != total_data.begin() + end_index; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins.uint64_feasigns_;
        //暂不支持多维度，打平老逻辑(有需要的时候加上)
        for (const auto feasign : feasign_v) {
          uint64_t cur_key = feasign.sign().uint64_feasign_;
          int shard_id = cur_key % thread_keys_shard_num_;
          this->thread_keys_[i][shard_id][0].insert(cur_key);
        }
      }
    };
  }
  std::vector<std::thread> threads;
  size_t len_per_thread = total_len / thread_keys_thread_num_;
  size_t remain = total_len % thread_keys_thread_num_;
  size_t begin = 0;
  for (size_t i = 0; i < (size_t)thread_keys_thread_num_; i++) {
    threads.push_back(
      std::thread(first_uniq_func, record_vec, begin,
        begin + len_per_thread + (i < remain ? 1 : 0), i));
    begin += len_per_thread + (i < remain ? 1 : 0);
  }
  for (auto& t : threads) {
    t.join();
  }
  threads.clear();
  timeline.Pause();
  auto step_1 = timeline.ElapsedSec();

  //step2 insert into together
  timeline.Start();
  auto merge_ins_func = [this, gpu_task](int shard_num, int dim_id) -> void {
    for (int i = 0; i < thread_keys_thread_num_; ++i) {
      gpu_task->batch_add_keys(shard_num, dim_id, thread_keys_[i][shard_num][dim_id]);
      thread_keys_[i][shard_num][dim_id].clear();
    }
  };
  for (int i = 0; i < thread_keys_shard_num_; ++i) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      threads.push_back(std::thread(merge_ins_func, i, j));
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  threads.clear();
  timeline.Pause();
  auto step_2 = timeline.ElapsedSec();

  //step3 精细化去重
  timeline.Start();
  gpu_task->unique_keys();
  timeline.Pause();
  auto step_3 = timeline.ElapsedSec();
  VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  PreBuildTask cost-detail  "
          << "rough-dedup: " << step_1
          << "s  add-batch: " << step_2
          << "s  unique_keys:" << step_3 << "s"; 
}

void PSGPUWrapper::BuildPull(std::shared_ptr<HeterContext> gpu_task) {

#if (defined PADDLE_WITH_PSLIB) || (defined PADDLE_WITH_PSCORE)
  platform::Timer timeline;
  timeline.Start();

#ifdef PADDLE_WITH_PSLIB
  auto fleet_ptr = FleetWrapper::GetInstance();
#else
  auto fleet_ptr = paddle::distributed::FleetWrapper::GetInstance();
#endif

#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  //设置日期，ps内部pull的时候需要根据day_id做decay
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
  //获取sparse-value的指针
  auto& pull_keys = gpu_task->feature_keys_;
  auto& pull_value = gpu_task->value_ptr_;
  auto pull_value_func = [this, &pull_keys, &pull_value, &fleet_ptr, &gpu_task](int i, int j) -> void {
    size_t key_size = pull_keys[i][j].size();
    pull_value[i][j].resize(key_size);
    int32_t cnt = 0;
    while (true) {
#ifdef PADDLE_WITH_PSLIB
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
          i, reinterpret_cast<char**>(pull_value[i][j].data()),
          this->table_id_, pull_keys[i][j].data(), key_size, gpu_task->pass_id_);
#else
      auto tt = fleet_ptr->worker_ptr_->PullSparsePtr(
          reinterpret_cast<char**>(pull_value[i][j].data()), this->table_id_,
          pull_keys[i][j].data(), key_size);
#endif
      tt.wait();
      int32_t status = -1;
      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status == 0) {
        break;
      } else {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }
    }
  };
  std::vector<std::future<void>> task_futures;
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      task_futures.emplace_back(pull_thread_pool_[i]->enqueue(pull_value_func, i, j));
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  timeline.Pause();
  fleet_ptr->pslib_ptr_->_worker_ptr->release_table_mutex(0);
  VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  build-pull-detail cost: " << timeline.ElapsedSec() << "s";
#endif
  if (multi_node_) {
    auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
    if (!gloo_wrapper->IsInitialized()) {
      VLOG(0) << "GLOO is not inited";
      gloo_wrapper->Init();
    }
    gloo_wrapper->Barrier();
  }
}

//里面用到了一些静态变量，同一时刻只允许有一个线程运行这个函数
void PSGPUWrapper::BuildGPUTask(std::shared_ptr<HeterContext> gpu_task) {
#define FOUR_VECTOR_INIT(vec, first_size, second_size, thrid_size) \
  vec.resize(first_size); \
  for (auto& iter1 : vec) { \
    iter1.resize(second_size); \
    for (auto& iter2 : iter1) { \
      iter2.resize(thrid_size); \
      for (auto& iter3 : iter2) { \
        iter3.clear(); \
      } \
    } \
  }
#define THIRD_VECTOR_INIT(vec, first_size, second_size, thrid_size) \
  vec.resize(first_size); \
  for (auto& iter1 : vec) { \
    iter1.resize(second_size); \
    for (auto& iter2 : iter1) { \
      iter2.resize(thrid_size); \
    } \
  }

  platform::Timer timeline;
  timeline.Start();
  std::vector<std::future<void>> task_futures;
  int device_num = heter_devices_.size();
  //step1 input: 分片->维度->key; => output: 分片->设备->维度->key
  auto& pull_keys = gpu_task->feature_keys_;
  auto& pull_values = gpu_task->value_ptr_;
  static std::vector<std::vector<std::vector<std::vector<FeatureKey>>>> s_first_keys;
#ifdef PADDLE_WITH_PSLIB
  static std::vector<std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>> s_first_value;
#elif PADDLE_WITH_PSCORE
  static std::vector<std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>> s_first_value;
#endif
  FOUR_VECTOR_INIT(s_first_keys, thread_keys_shard_num_, device_num, multi_mf_dim_)
  FOUR_VECTOR_INIT(s_first_value, thread_keys_shard_num_, device_num, multi_mf_dim_)
  auto& l_first_keys = s_first_keys;
  auto& l_first_value = s_first_value;
  auto func_first = [this, &pull_keys, &pull_values, &l_first_keys, &l_first_value, &device_num](int shard_id, int dim_id) -> void {
    auto& l_keys = pull_keys[shard_id][dim_id];
    auto& l_values = pull_values[shard_id][dim_id];
    for (size_t i = 0; i < l_keys.size(); i++) {
      int dev_id = l_keys[i] % device_num;
      l_first_keys[shard_id][dev_id][dim_id].push_back(l_keys[i]);
      l_first_value[shard_id][dev_id][dim_id].push_back(l_values[i]);
    }
  };
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(func_first, i, j));
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();
  timeline.Pause();
  auto step_1 = timeline.ElapsedSec();
  timeline.Start();

  //step2 通过前缀计算，获得结果大小 prefix_sum: 设备->维度->分片数量
  //      l_second_key/s_second_value: 设备->维度->具体的值
  static std::vector<std::vector<std::vector<int>>> prefix_sum;
#ifdef PADDLE_WITH_PSLIB
  static std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>> s_second_value;
#elif PADDLE_WITH_PSCORE
  static std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>> s_second_value;
#endif
  THIRD_VECTOR_INIT(prefix_sum, device_num, multi_mf_dim_, thread_keys_shard_num_ + 1)
  THIRD_VECTOR_INIT(s_second_value, device_num, multi_mf_dim_, 0)
  auto& l_second_key = gpu_task->device_keys_;
  auto& l_second_value = s_second_value;
  auto& l_prefix_sum = prefix_sum;
  auto func_second = [this, &l_prefix_sum, &l_first_keys, &l_second_key, &l_second_value]
                      (int device_id, int dim_id) -> void {
    l_prefix_sum[device_id][dim_id][0] = 0;
    for (int j = 0; j < this->thread_keys_shard_num_; j++) {
      l_prefix_sum[device_id][dim_id][j+1] = l_prefix_sum[device_id][dim_id][j] + l_first_keys[j][device_id][dim_id].size();
    }
    l_second_key[device_id][dim_id].resize(l_prefix_sum[device_id][dim_id][this->thread_keys_shard_num_]);
    l_second_value[device_id][dim_id].resize(l_prefix_sum[device_id][dim_id][this->thread_keys_shard_num_]);
  };
  for (int i = 0; i < device_num; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(func_second, i, j));
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();
  timeline.Pause();
  auto step_2 = timeline.ElapsedSec();
  timeline.Start();

  //step3, 具体的key/value => 转入到l_second_key/l_second_value
  //       这就是要最终转入到gpu中key-value分设备后的cpu数据了
  auto func_third = [this, &l_second_key, &l_second_value, &l_first_keys, &l_first_value, &l_prefix_sum]
                    (int shard_id, int device_id, int dim_id) -> void {
    auto& input_key = l_first_keys[shard_id][device_id][dim_id];
    auto& input_value = l_first_value[shard_id][device_id][dim_id];
    auto& output_key = l_second_key[device_id][dim_id];
    auto& output_value = l_second_value[device_id][dim_id];
    int start_index = prefix_sum[device_id][dim_id][shard_id];
    for (size_t i = 0; i < input_key.size(); i++) {
      output_key[i + start_index] = input_key[i];
      output_value[i + start_index] = input_value[i];
    }
  };
  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < device_num; j++) {
      for (int k = 0; k < multi_mf_dim_; k++) {
        task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(func_third, i, j, k));
      }
    }
  }
  for (auto& f : task_futures) {
    f.wait();
  }
  task_futures.clear();
  timeline.Pause();
  auto step_3 = timeline.ElapsedSec();
  timeline.Start();

  //step4 初始化gpu-table相关数据了
  size_t size_max = 0;
  for (int i = 0; i < device_num; i++) {
    size_t tmp_size = 0;
    for (int j = 0; j < multi_mf_dim_; j++) {
      tmp_size += l_second_key[i][j].size();
    }
    size_max = std::max(size_max, tmp_size);
  }
  if (size_max <= 0) {
    VLOG(0) << "Skip build gpu ps cause feasign nums = " << size_max;
    return;
  }
  if (HeterPs_) {
    delete HeterPs_;
    HeterPs_ = nullptr;
  }
  HeterPs_ = HeterPsBase::get_instance(size_max, resource_, ps_accessor_type_, gpu_value_type_);
  CHECK(HeterPs_ != nullptr);
  HeterPs_->set_nccl_comm_and_size(inner_comms_, inter_comms_, node_size_);
  HeterPs_->set_multi_mf_dim(max_mf_dim_);

  //step5, cpu数据转化到gpu数据，并构造gpu-table了
  auto transfor_value_obj = g_transfor;
  auto build_table_func = [this, &l_second_key, &l_second_value, &transfor_value_obj] 
                          (int device_id, int dim_id) -> void {
    int cur_dim_size = this->index_dim_vec_[dim_id];
    size_t gpu_value_size = transfor_value_obj->get_gpu_value_size(cur_dim_size);
    auto& cpu_keys = l_second_key[device_id][dim_id];
    auto& cpu_values = l_second_value[device_id][dim_id];
    size_t keys_len = cpu_keys.size();
    this->mem_pools_[device_id * this->multi_mf_dim_ + dim_id] = new MemoryPool(keys_len, gpu_value_size);
    auto& mem_pool = this->mem_pools_[device_id * this->multi_mf_dim_ + dim_id];
    for (size_t k = 0; k < keys_len; k++) {
      void* to_value_ptr = mem_pool->mem_address(k);
      transfor_value_obj->value_cpu_to_gpu(cpu_values[k], to_value_ptr, cur_dim_size);
    }
    auto device_index = resource_->dev_id(device_id);
    platform::CUDADeviceGuard guard(device_index);

    this->hbm_pools_[device_id * this->multi_mf_dim_ + dim_id] = new HBMMemoryPool(mem_pool);
    auto& cur_pool = this->hbm_pools_[device_id * this->multi_mf_dim_ + dim_id];
    this->HeterPs_->build_ps(device_id, cpu_keys.data(), cur_pool->mem(), keys_len,
                             gpu_value_size, 500000, 2);
    delete mem_pool;
  };
  
  std::vector<std::thread> threads;
  threads.resize(device_num * multi_mf_dim_);
  for (int i = 0; i < device_num; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      threads[i + j * device_num] = std::thread(build_table_func, i, j);
    }
  }
  
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  auto step_4 = timeline.ElapsedSec();
  VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  build_gpu_table_detail  "
          << "  device_picec_a:" << step_1
          << "s  device_picec_b:" << step_2
          << "s  device_picec_c:" << step_3
          << "s  trans_to_gpu:" << step_4 << "s";

#undef FOUR_VECTOR_INIT
#undef THIRD_VECTOR_INIT
}

void PSGPUWrapper::LoadIntoMemory(bool is_shuffle) {
  platform::Timer timer;
  timer.Start();
  dataset_->LoadIntoMemory();
  timer.Pause();
  auto load_s = timer.ElapsedSec();
  timer.Start();
  if (is_shuffle) {
    dataset_->LocalShuffle();
  }
  timer.Pause();
  auto shuffle_s = timer.ElapsedSec();
  
  InitSlotInfo();
  std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
  gpu_task->pass_id_ = (uint16_t)(dataset_->GetPassID());
  VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  LoadIntoMemory cost: " << load_s << "s  Shuffle cost:" << shuffle_s << "s";

  dataset_mutex_.lock();
  dataset_pipe_.push(dataset_);
  dataset_mutex_.unlock();
  data_ready_channel_->Put(gpu_task);
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
    if (!data_ready_channel_->Get(gpu_task)) {
      continue;
    }
    platform::Timer timer;
    timer.Start();
    PreBuildTask(gpu_task);
    timer.Pause();
    VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  PreBuildTask cost: " << timer.ElapsedSec() << "s";
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
    platform::Timer timer;
    timer.Start();
    BuildPull(gpu_task);
    timer.Pause();
    VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  BuildPull cost: " << timer.ElapsedSec() << "s";
    buildpull_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}

void PSGPUWrapper::build_task() {
  std::shared_ptr<HeterContext> gpu_task = nullptr;
  // train end, gpu free
  if (!gpu_free_channel_->Get(gpu_task)) {
    return;
  }
  // ins and pre_build end
  if (!buildpull_ready_channel_->Get(gpu_task)) {
    return;
  }
  platform::Timer timer;
  timer.Start();
  BuildGPUTask(gpu_task);
  timer.Pause();
  VLOG(0) << "pass_id:" << gpu_task->pass_id_ << "  build_gpu_table cost: " << timer.ElapsedSec() << "s";
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
  VLOG(0) << "pass_id:" << current_task_->pass_id_ << "  begin_pass cost: " << timer.ElapsedSec() << "s";
}



void PSGPUWrapper::EndPass() {
  if (!current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[EndPass] current task has been ended."));
  }
  auto fleet_ptr = FleetWrapper::GetInstance();
  platform::Timer timer;
  timer.Start();
  

  int thread_num = 8;
  auto dump_pool_to_cpu_func = [this, thread_num](int i, int j, int z) {

    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(this->resource_->dev_id(i)));

    auto& hbm_pool = this->hbm_pools_[i * this->multi_mf_dim_ + j];
    auto& device_keys = this->current_task_->device_keys_[i][j];
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
    size_t feature_value_size = g_transfor->get_gpu_value_size(mf_dim);
    auto cpu_value =  memory::Alloc(phi::GPUPinnedPlace(), feature_value_size * real_len);
    char* cpu_value_ptr = reinterpret_cast<char*>(cpu_value->ptr());
    uint64_t offset = left * feature_value_size;
    gpuStream_t streams;
    cudaStreamCreate(&streams);
    cudaMemcpyAsync(cpu_value_ptr, hbm_pool->mem() + offset,
               feature_value_size * real_len, cudaMemcpyDeviceToHost, streams);
    cudaStreamSynchronize(streams);
    cudaStreamDestroy(streams);
    CHECK(len == hbm_pool->capacity());
    uint64_t unuse_key = std::numeric_limits<uint64_t>::max();
    for (int i = left; i < right; ++i) {
      if (device_keys[i] == unuse_key) {
        continue;
      }
      size_t local_offset = (i - left) * feature_value_size;
      void* gpu_val = (void*)(cpu_value_ptr + local_offset);
      g_transfor->value_gpu_to_cpu(gpu_val);
    }
  };

  size_t device_num = heter_devices_.size();
  std::vector<std::thread> threads(device_num * multi_mf_dim_ * thread_num);
  for (size_t i = 0; i < device_num; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      for (int k = 0; k < thread_num; k++) {
        threads[(i + j * device_num) * thread_num + k] =
            std::thread(dump_pool_to_cpu_func, i, j, k);
      }
    }
  }
  for (std::thread& t : threads) {
    t.join();
  }

  for (size_t i = 0; i < hbm_pools_.size(); i++) {
    delete hbm_pools_[i];
  }
  gpu_task_pool_.Push(current_task_);
  current_task_ = nullptr;
  gpu_free_channel_->Put(current_task_);
  timer.Pause();
  VLOG(0) << "EndPass end, cost time: " << timer.ElapsedSec() << "s";
}

//拉取gpu-table数据，并做cvm填充(对应pull_box_sparse_op算子, 非动态维度)
void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const int hidden_size) {
  
  const std::vector<int> slot_dim;
  PullSparse(place, table_id, keys, values, slot_lengths, slot_dim, hidden_size);
  return;
}

//拉取gpu-table数据，并做cvm填充(对应pull_gpups_sparse_op算子, 非动态维度)
void PSGPUWrapper::PullSparse(
    const paddle::platform::Place& place, const int table_id,
    const std::vector<const uint64_t*>& keys, const std::vector<float*>& values,
    const std::vector<int64_t>& slot_lengths,
    const std::vector<int>& slot_dim,  // dimension for each slot
    const int hidden_size) {
    
  if (!platform::is_gpu_place(place)) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
      "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  //step1 将散乱的key重整到一块连续的空间上面来
  size_t total_length = std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  size_t feature_value_size = g_transfor->get_gpu_value_size(max_mf_dim_);
  auto buf = memory::Alloc(place, total_length * feature_value_size);
  void* total_values_gpu = reinterpret_cast<void*>(buf->ptr());
  int device_id = place.GetDeviceId();
  int devid_2_index = HeterPs_->get_index_by_devid(device_id);
  LoDTensor& total_keys_tensor = keys_tensor_[devid_2_index];
  uint64_t* total_keys = reinterpret_cast<uint64_t*>(total_keys_tensor.mutable_data<int64_t>(
                                {int64_t(total_length), 1}, place));
  auto slot_lengths_lod = slot_lengths;
  for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }

  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();

  PinnedVector pinned_buf_key(keys.data(), keys.size() * sizeof(uint64_t*), stream, place);
  uint64_t** gpu_keys = pinned_buf_key.get_gpu_ptr<uint64_t*>();
  PinnedVector pinned_buf_length(slot_lengths_lod.data(), slot_lengths.size() * sizeof(int64_t), stream, place);
  int64_t* gpu_len = pinned_buf_length.get_gpu_ptr<int64_t>();

  this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                  static_cast<int>(slot_lengths.size()),
                  static_cast<int>(total_length));
  
  //step2 查表获得gpu-value数据
  HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu, total_length);

  //step3 做cvm转换处理
  PinnedVector pinned_buf_value(values.data(), values.size() * sizeof(float*), stream, place);
  float** gpu_values = pinned_buf_value.get_gpu_ptr<float*>();
  if (slot_dim.size() != 0) { //动态mf模式
    PinnedVector pinned_dim(slot_dim.data(), slot_dim.size() * sizeof(int), stream, place);
    int* gpu_dim = pinned_dim.get_gpu_ptr<int>();
    g_transfor->value_to_cvm(gpu_values, total_values_gpu, gpu_keys, slot_lengths.size(), gpu_len,
      gpu_dim, total_length, 0, feature_value_size, stream);
  } else {
    g_transfor->value_to_cvm(gpu_values, total_values_gpu, gpu_keys, slot_lengths.size(), gpu_len,
        nullptr, total_length, hidden_size, feature_value_size, stream);
  }
}

void PSGPUWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                  const int table_id,
                                  const std::vector<const uint64_t*>& keys,
                                  const std::vector<const float*>& grad_values,
                                  const std::vector<int64_t>& slot_lengths,
                                  const int hidden_size, const int batch_size) {
  if (!platform::is_gpu_place(place)) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  //step1 将零散的梯度重整到一块连续的空间上面来
  int64_t total_length = std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  size_t grad_value_size = g_transfor->get_gpu_push_value_size(max_mf_dim_);
  int device_id = place.GetDeviceId();
  int devid_2_index = HeterPs_->get_index_by_devid(device_id);
  LoDTensor& cached_total_keys_tensor = keys_tensor_[devid_2_index];
  uint64_t* total_keys = reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  
  PinnedVector pinned_gpu_values(grad_values.data(), grad_values.size() * sizeof(float*), stream, place);
  float** gpu_values = pinned_gpu_values.get_gpu_ptr<float*>();

  PinnedVector pinned_slot_vector(slot_vector_.data(), slot_vector_.size() * sizeof(int), stream, place);
  int* d_slot_vector = pinned_slot_vector.get_gpu_ptr<int>();

  auto slot_lengths_lod = slot_lengths;
  for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  PinnedVector pinned_buf_length(slot_lengths_lod.data(), slot_lengths_lod.size() * sizeof(int64_t), stream, place);
  int64_t* gpu_len = pinned_buf_length.get_gpu_ptr<int64_t>();

  auto buf = memory::Alloc(place, total_length * grad_value_size);
  void* total_grad_values_gpu = reinterpret_cast<void*>(buf->ptr());
      
  if (hidden_size == 0) { //动态mf
    PinnedVector pinned_mf_dim(slot_mf_dim_vector_.data(), slot_mf_dim_vector_.size() * sizeof(int), stream, place);
    int* d_mf_dim_vector = pinned_mf_dim.get_gpu_ptr<int>();
    g_transfor->grad_to_push(
          total_grad_values_gpu, gpu_values, slot_lengths.size(), gpu_len, d_mf_dim_vector,
          total_length, 0, grad_value_size, batch_size, d_slot_vector, stream);
  } else {
    g_transfor->grad_to_push(
          total_grad_values_gpu, gpu_values, slot_lengths.size(), gpu_len, 0,
          total_length, hidden_size, grad_value_size, batch_size, d_slot_vector, stream);
  }

  //step2，梯度更新了
  HeterPs_->push_sparse(devid_2_index, total_keys, total_grad_values_gpu,
                          static_cast<int>(total_length));
}

}  // end namespace framework
}  // end namespace paddle
#endif
