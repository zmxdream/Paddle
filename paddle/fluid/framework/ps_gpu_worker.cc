/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mutex>

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#if (defined PADDLE_WITH_NCCL || defined PADDLE_WITH_RCCL) && \
    (defined PADDLE_WITH_PSLIB)
#include "paddle/fluid/platform/cuda_device_guard.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

std::atomic<int> PSGPUWorker::shape_check_count_(16);
std::atomic<bool> PSGPUWorker::shape_check_flag_(true);

void PSGPUWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  this->HogwildWorker::CreateDeviceResource(main_prog);
  if (scope_num_ != 1) {
    auto& block = main_prog.Block(0);
    for (int i = 0; i < scope_num_; i++) {
      auto thread_tmp = &thread_scope_->NewScope();
      thread_scope_vec_.push_back(thread_tmp);
    }
    for (auto& scope : thread_scope_vec_) {
      for (auto& var : block.AllVars()) {
        std::string name = var->Name();
        if (!var->Persistable()) {
          auto* ptr = scope->Var(var->Name());
          InitializeVariable(ptr, var->GetType());
        }
      }
    }
    // 设置为Runtime infer shape flag
    for (auto& op : ops_) {
      op->SetIsRuntimeInferShape(true);
    }

    // reusing memory
    auto input_names = device_reader_->GetInputVarNames();
    std::set<std::string> input_names_set(input_names.begin(), input_names.end());
    for (auto& scope : thread_scope_vec_) {
      std::vector<Variable*> need_reuse;
      for (auto& var : block.AllVars()) {
        std::string name = var->Name();
        if (!var->Persistable()) {
          if (input_names_set.find(var->Name()) != input_names_set.end()) {
            continue;
          }
          auto* ptr = scope->FindLocalVar(var->Name());
          PADDLE_ENFORCE_NE(ptr, nullptr,
                phi::errors::NotFound("The var %s is not found.", var->Name()));
          need_reuse.push_back(ptr);
        }
      }
      need_reuse_var_vec_[scope] = std::move(need_reuse);
    }
    {
      need_reuse_var_.clear();
      for (auto& var : block.AllVars()) {
        std::string name = var->Name();
        if (!var->Persistable()) {
          if (input_names_set.find(var->Name()) != input_names_set.end()) {
            continue;
          }
          auto* ptr = thread_scope_->FindLocalVar(var->Name());
          PADDLE_ENFORCE_NE(ptr, nullptr,
                phi::errors::NotFound("The var %s is not found.", var->Name()));
          need_reuse_var_.push_back(ptr);
        }
      }
    }
  }
}

void PSGPUWorker::BindingDataFeedMemory() {
  if (scope_num_ == 1) {
    this->HogwildWorker::BindingDataFeedMemory();
  } else {
    for (auto& scope : thread_scope_vec_) {
      // data feed 设置每个scope的feed_vec...不过不都是空的吗
      device_reader_->AssignFeedVar(*scope);
    }
  }
}

void PSGPUWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  mpi_rank_ = desc.mpi_rank();
  trainer_desc_ = desc;
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    uint64_t table_id =
        static_cast<uint64_t>(param_.sparse_table(i).table_id());
    TableParameter table = param_.sparse_table(i);
    sparse_key_names_[table_id].resize(table.sparse_key_name_size());
    for (int j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_names_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_names_[table_id].resize(table.sparse_value_name_size());
    for (int j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_names_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_names_[table_id].resize(table.sparse_grad_name_size());
    for (int j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_names_[table_id][j] = table.sparse_grad_name(j);
    }
    label_var_name_[table_id] = table.label_var_name();
    sparse_push_keys_[table_id] = std::vector<uint64_t>();
  }

  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_value_names_[table_id].resize(table.dense_value_name_size());
    for (int j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_names_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    stat_var_name_map_[param_.stat_var_names(i)] = 1;
  }

  need_to_push_sparse_ = param_.push_sparse();
  need_to_push_dense_ = param_.push_dense();

  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  for (int i = 0; i < desc.check_nan_var_names_size(); ++i) {
    check_nan_var_names_.push_back(desc.check_nan_var_names(i));
  }
  copy_table_config_ = desc.copy_table_config();
  for (int i = 0; i < copy_table_config_.src_sparse_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_sparse_tables(i);
    uint64_t dest_table = copy_table_config_.dest_sparse_tables(i);
    VLOG(3) << "copy_sparse_tables_ push back " << src_table << "->"
            << dest_table;
    copy_sparse_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (auto& m : copy_table_config_.table_denpendency_map()) {
    if (sparse_key_names_.find(m.key()) != sparse_key_names_.end()) {
      // currently only support one dependency
      for (auto& value : m.values()) {
        table_dependency_[m.key()] = value;
      }
    }
  }
}

void PSGPUWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void PSGPUWorker::PrepareCudaGraph() {
  std::string enable_cuda_graph_capture_attr_name = "enable_cuda_graph_capture";
  op_or_cudagraphs_.reserve(ops_.size());

  static const std::unordered_set<std::string> op_whitelist = {
    "adam",
    "coalesce_tensor",
  };
  // these op can not be captured
  static const std::unordered_set<std::string> op_blacklist = {
    "c_sync_calc_stream",
    "c_allreduce_sum",
    "c_sync_comm_stream",
  };
  // when op is captured, its inputs and outputs and their grads will be never changed
  // so the capture attribute can infect another op whose all inputs and outputs nerver changed
  std::unordered_set<std::string> var_whitelist;
  for (auto& op : ops_) {
    if (op_whitelist.find(op->Type()) != op_whitelist.end() ||
        (op->HasAttr(enable_cuda_graph_capture_attr_name) && op->Attr<int>(enable_cuda_graph_capture_attr_name))) {
      for (auto& input : op->InputVars()) {
        var_whitelist.emplace(input);
        var_whitelist.emplace(framework::GradVarName(input));
      }
      for (auto& output : op->OutputVars(true)) {
        var_whitelist.emplace(output);
        var_whitelist.emplace(framework::GradVarName(output));
      }
    }
  }

  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      bool need_capture = false;
      // if (op_blacklist.find(op->Type()) == op_blacklist.end()) {
      //   if (op->HasAttr(enable_cuda_graph_capture_attr_name) && op->Attr<int>(enable_cuda_graph_capture_attr_name)) {
      //     need_capture = true;
      //   }
      //   if (!need_capture) {
      //     need_capture = true;
      //     for (auto& input : op->InputVars()) {
      //       if (var_whitelist.find(input) == var_whitelist.end()) {
      //         need_capture = false;
      //         break;
      //       }
      //     }
      //     if (need_capture) {
      //       for (auto& output : op->OutputVars(true)) {
      //         if (var_whitelist.find(output) == var_whitelist.end()) {
      //           need_capture = false;
      //           break;
      //         }
      //       }
      //     }
      //   }
      // }

      if (op_or_cudagraphs_.empty() || op_or_cudagraphs_.back().need_capture != need_capture) {
        op_or_cudagraphs_.emplace_back();
        op_or_cudagraphs_.back().need_capture = need_capture;
      }
      auto& op_or_cuda_graph = op_or_cudagraphs_.back();
      if (need_capture) {
        if (op_or_cuda_graph.name.empty()) {
          op_or_cuda_graph.name = "cuda_graph:";
        } else {
          op_or_cuda_graph.name += ":";
        }
        op_or_cuda_graph.name += op->Type();
      }
      op_or_cuda_graph.ops.emplace_back(op);
    }
  }
}

PSGPUWorker::~PSGPUWorker() {
  stop_token_.store(true);
  for (auto& thread : task_threads_) {
    if (thread.joinable()) {
        thread.join();
    }
  }
}

int PSGPUWorker::OpRunAndShapeCheck(OperatorBase& op,
                                    const Scope& scope,
                                    const platform::Place& place) {
    if (shape_check_flag_.load()) {
      // before op run
      InferShapeCheckData check_data;
      auto& pre_dims = check_data.pre_dims;
      auto& pre_lods = check_data.pre_lods;
      auto& after_dims = check_data.after_dims;
      auto& after_lods = check_data.after_lods;
      RuntimeContext ctx(op.Inputs(), op.Outputs(), scope);
      RuntimeInferShapeContext infer_shape_ctx(op, ctx, scope);
      auto outnames = op.Outputs();
      for (auto& var_name_item : outnames) {
        pre_dims.push_back(infer_shape_ctx.GetOutputsDim(var_name_item.first));
        pre_lods.push_back(infer_shape_ctx.GetOutputsLod(var_name_item.first));
      }

      // op run
      op.Run(scope, place);

      // after op run
      for (auto& var_name_item : outnames) {
        after_dims.push_back(infer_shape_ctx.GetOutputsDim(var_name_item.first));
        after_lods.push_back(infer_shape_ctx.GetOutputsLod(var_name_item.first));
      }

      std::string op_name = "unknow_op";
      if (op.Info().HasOpProtoAndChecker()) {
        op_name = op.Info().Proto().type();
      }

      #define SHAPE_CHECK_EQ(__VAL0, __VAL1) \
          PADDLE_ENFORCE_EQ(__VAL0, __VAL1, platform::errors::Fatal( \
                              "Shape check dims/lods error, op name: %s .", op_name))

      SHAPE_CHECK_EQ(pre_dims.size(), after_dims.size());
      for (size_t i = 0; i < pre_dims.size(); i++) {
        SHAPE_CHECK_EQ(pre_dims[i].size(), after_dims[i].size());
        for (size_t j = 0; j < pre_dims[i].size(); j++) {
          SHAPE_CHECK_EQ(pre_dims[i][j], after_dims[i][j]);
        }
      }

      SHAPE_CHECK_EQ(pre_lods.size(), after_lods.size());
      for (size_t i = 0; i < pre_lods.size(); i++) {
        SHAPE_CHECK_EQ(pre_lods[i].size(), after_lods[i].size());
        for (size_t j = 0; j < pre_lods[i].size(); j++) {
          auto& x = pre_lods[i][j];
          auto& y = after_lods[i][j];
          SHAPE_CHECK_EQ(x.size(), y.size());
          for (size_t i = 0; i < x.size(); i++) {
            const auto &x_level = x[i];
            const auto &y_level = y[i];
            SHAPE_CHECK_EQ(x_level.size(), y_level.size());
            for (size_t j = 0; j < x_level.size(); j++) {
              SHAPE_CHECK_EQ(x_level[j], y_level[j]);
            }
          }
        }
      }
      #undef SHAPE_CHECK_EQ
    } else {
       op.Run(scope, place);
    }
    return 0;
}


void PSGPUWorker::TrainFiles() {

  VLOG(0) << "Begin to train files";
  platform::SetNumThreads(1);
  platform::Timer timeline;
  timeline.Start();

  int total_ins_num = 0;

  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch;
  int batch_cnt = 0;
  int graph_batch_size = 0;
  platform::SetDeviceId(place_.GetDeviceId());


  // async infershape
  pack_is_end_.store(false);

  // scope_num_ != 1是什么情况
  if (scope_num_ != 1) {

    //每个scope一个task 
    for (size_t i = 0; i < thread_scope_vec_.size(); i++) {
      TaskData task;
      task.scope = thread_scope_vec_[i];
      free_task_queue_.Push(task);
    }

    // async infershape
    //int task_threads_num_ {6};
    //int scope_num_ {task_threads_num_ + 1};
    // 
    thread_count_.store(task_threads_num_);
    task_threads_.reserve(task_threads_num_);

    // 开启多个线程
    for (int i = 0; i < task_threads_num_; i++) {

      task_threads_.emplace_back(std::thread([this]() -> void {

        while (true) {

          // 拿到一个pack
          auto pack = device_reader_->get_pack(nullptr);

          if (pack == nullptr) {
            int thread_num = thread_count_.fetch_sub(1);
            if (thread_num == 1) {
              pack_is_end_.store(true);
            }
            return;
          }

          auto task = free_task_queue_.Pop();
          task.pack = pack;
          task.ins_num = pack->ins_num();
          device_reader_->PackToScope(task.pack, task.scope);

          for (size_t i = 0; i < ops_.size(); i++) {
            auto& op = ops_[i];
            bool need_skip = false;
            for (auto t = 0u; t < skip_ops_.size(); ++t) {
              if (op->Type().find(skip_ops_[t]) != std::string::npos) {
                need_skip = true;
                break;
              }
            }
            if (!need_skip) {
              op->RuntimeInferShape(*task.scope);
            }
          }

          using_task_queue_.Push(task);

        }


      }));

    }

  }

  while (true) {

    auto thread_scope = thread_scope_;

    TaskData cur_task;

    if (scope_num_ == 1) {

      cur_batch = device_reader_->Next();

    } else {

      while (true) {
        if (using_task_queue_.Size() != 0) {

          cur_task = using_task_queue_.Pop();
          cur_batch = cur_task.ins_num;
          break;

        }
        bool is_end = pack_is_end_.load();
        if (is_end) {
          if (using_task_queue_.Size() == 0) {
            cur_batch = 0;
            break;
          }
        }
        std::this_thread::sleep_for(
          std::chrono::microseconds(200));
      }
      thread_scope = cur_task.scope;
      // tensor share buffer
      std::vector<Variable*>& cur_scope_vars = need_reuse_var_vec_[thread_scope];
      PADDLE_ENFORCE_EQ(cur_scope_vars.size(), need_reuse_var_.size(),
                        platform::errors::Fatal(
                              "reuse vars size must be same."));
      for (size_t i = 0; i < need_reuse_var_.size(); i++) {
        Variable* child = cur_scope_vars[i];
        Variable* parent = need_reuse_var_[i];
        if (child->IsType<LoDTensor>()) {
          child->GetMutable<LoDTensor>()->ShareBufferWith(*(parent->GetMutable<LoDTensor>()));
        }
      }
    }

    if (cur_batch <= 0) {
      break;
    }

    total_ins_num += cur_batch;


    // 这个是干啥的
    //
    if (shape_check_flag_.load()) {
      VLOG(0) << "Begin OpRunAndShapeCheck, "
            << shape_check_count_.load();
      if (scope_num_ == 1 || shape_check_count_.fetch_sub(1) <= 0) {
        VLOG(0) << "End OpRunAndShapeCheck."
            << shape_check_count_.load();
        shape_check_flag_ = false;
      }
    }

    if (op_or_cudagraphs_.empty()) {
      // first batch we run original ops to check whethere the tensors has lod
      for (auto& op : ops_) {
        bool need_skip = false;
        for (auto t = 0u; t < skip_ops_.size(); ++t) {
          if (op->Type().find(skip_ops_[t]) != std::string::npos) {
            need_skip = true;
            break;
          }
        }
        if (!need_skip) {
/*
       std::vector<std::string> check_nan_var_names {"fc_news_u1.tmp_0", "fc_news_u1.tmp_1", "_generated_var_6", "_generated_var_0"};

    for (std::string& var_name : check_nan_var_names) {
      Variable* var = thread_scope->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr || !tensor->IsInitialized()) {
        continue;
      }
      if (framework::TensorContainsInf(*tensor) ||
          framework::TensorContainsNAN(*tensor)) {
        static std::mutex mutex;
        {
          std::lock_guard<std::mutex> lock(mutex);
          VLOG(0) << "worker " << thread_id_ << ": " << var_name
                  << " cantains inf or nan";
          auto all_vars = thread_scope->LocalVarNames();
          std::stringstream ss;
          ss << "====== worker " << thread_id_ << "======\n";
          for (auto& local_var : all_vars) {
            platform::PrintVar(thread_scope, local_var, local_var, &ss);
            ss << "\n";
          }
          std::cout << ss.str() << std::endl;
          VLOG(0) << "worker " << thread_id_ << "print nan var done....";
        }
        sleep(600);
        exit(-1);
      }
    }
*/

//           VLOG(0) << "debug run op:" << op->Type(); 
/*
          if (op->Type() == "weighted_random_sample") {

              // check the output of data_norm
              // Variable* var = thread_scope->FindVar("_generated_var_6");
              Variable* var = thread_scope->FindVar("fc_news_u1.tmp_1");
              if (var == nullptr) {
                  continue;
              }

              LoDTensor* tensor = var->GetMutable<LoDTensor>();
              if (tensor == nullptr || !tensor->IsInitialized()) {
                 continue;
              }

              float* tensor_data = tensor->data<float>();
              const size_t total_len = tensor->dims()[0] * tensor->dims()[1];
              float  tmp_data[total_len];
              cudaMemcpy(tmp_data, tensor_data, total_len * sizeof(float), cudaMemcpyDeviceToHost);
              for (int i = 0; i < tensor->dims()[0]; i++) {
                  for (int j = 0; j < tensor->dims()[1]; j++) { 
                    std::cout << tmp_data[i * tensor->dims()[1] + j] << " ";
                  }
                  std::cout << std::endl;
              }
          }
*/
          OpRunAndShapeCheck(*op, *thread_scope, place_);

        }
      }
      graph_batch_size = cur_batch;
      PrepareCudaGraph();
    } else if (graph_batch_size != cur_batch || batch_cnt <= thread_id_) {
      // when batch_size changed, run original ops
      for (auto& op : ops_) {
        bool need_skip = false;
        for (auto t = 0u; t < skip_ops_.size(); ++t) {
          if (op->Type().find(skip_ops_[t]) != std::string::npos) {
            need_skip = true;
            break;
          }
        }
        if (!need_skip) {
          OpRunAndShapeCheck(*op, *thread_scope, place_);
        }
      }
    } else {
      // secend batch we capture the cudagraph
      for (auto& op_or_cuda_graph : op_or_cudagraphs_) {
        if (op_or_cuda_graph.need_capture) {
          if (op_or_cuda_graph.cudagraph == nullptr) {
            static std::mutex _capture_mutex;
            std::lock_guard<std::mutex> lock(_capture_mutex);
            platform::BeginCUDAGraphCapture(place_, cudaStreamCaptureModeThreadLocal);
            for (auto& op : op_or_cuda_graph.ops) {
              OpRunAndShapeCheck(*op, *thread_scope, place_);
            }
            op_or_cuda_graph.cudagraph = platform::EndCUDAGraphCapture();
          }

          platform::RecordEvent op_type_record_event(
              op_or_cuda_graph.name, platform::TracerEventType::Operator, 1);
          op_or_cuda_graph.cudagraph->Replay();
        } else {
          for (auto& op : op_or_cuda_graph.ops) {
            OpRunAndShapeCheck(*op, *thread_scope, place_);
          }
        }
      }
    }
    if (need_dump_field_) {
      DumpField(*thread_scope, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope, batch_cnt);
    }

    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = thread_scope->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr || !tensor->IsInitialized()) {
        continue;
      }
      if (framework::TensorContainsInf(*tensor) ||
          framework::TensorContainsNAN(*tensor)) {
        static std::mutex mutex;
        {
          std::lock_guard<std::mutex> lock(mutex);
          VLOG(0) << "worker " << thread_id_ << ": " << var_name
                  << " cantains inf or nan";
          auto all_vars = thread_scope->LocalVarNames();
          std::stringstream ss;
          ss << "====== worker " << thread_id_ << "======\n";
          for (auto& local_var : all_vars) {
            platform::PrintVar(thread_scope, local_var, local_var, &ss);
            ss << "\n";
          }
          std::cout << ss.str() << std::endl;
          VLOG(0) << "worker " << thread_id_ << "print nan var done....";
        }
        sleep(600);
        exit(-1);
      }
    }

    dev_ctx_->Wait();
    PrintFetchVars();
    thread_scope->DropKids();
    ++batch_cnt;

    if (scope_num_ != 1) {
      std::vector<Variable*>& cur_scope_vars = need_reuse_var_vec_[thread_scope];
      PADDLE_ENFORCE_EQ(cur_scope_vars.size(), need_reuse_var_.size(),
                        platform::errors::Fatal(
                              "reuse vars size must be same."));
      for (size_t i = 0; i < need_reuse_var_.size(); i++) {
        Variable* child = cur_scope_vars[i];
        Variable* parent = need_reuse_var_[i];
        if (child->IsType<LoDTensor>()) {
          parent->GetMutable<LoDTensor>()->ShareBufferWith(*(child->GetMutable<LoDTensor>()));
        }
      }
      device_reader_->get_pack(cur_task.pack);
      free_task_queue_.Push(cur_task);
    }
  }
  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs worker " << thread_id_ << " train cost "
          << timeline.ElapsedSec() << " seconds, ins_num: " << total_ins_num;
  return;
}

void PSGPUWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
  VLOG(0) << "Begin to train files with profiler";
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      op_name.push_back(op->Type());
    }
  }

  VLOG(3) << "op name size: " << op_name.size();
  op_total_time.resize(op_name.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int total_ins_num = 0;
  int cur_batch;
  timeline.Start();
  platform::SetDeviceId(place_.GetDeviceId());
  while ((cur_batch = device_reader_->Next()) > 0) {
    total_ins_num += cur_batch;
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    int run_op_idx = 0;
    dev_ctx_->Wait();
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        timeline.Start();
        VLOG(3) << "Going to run op " << op_name[run_op_idx];
        op->Run(*thread_scope_, place_);
        dev_ctx_->Wait();
        VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }
    timeline.Start();
    PrintFetchVars();
    thread_scope_->DropKids();
    dev_ctx_->Wait();
    timeline.Pause();
    total_time += timeline.ElapsedSec();
    timeline.Start();
  }
  VLOG(0) << "GpuPs worker " << thread_id_ << " train cost " << total_time
          << " seconds, ins_num: " << total_ins_num;
  for (size_t i = 0; i < op_name.size(); ++i) {
    VLOG(0) << "card:" << thread_id_ << ", op: " << op_name[i]
            << ", mean time: " << op_total_time[i] / total_ins_num
            << "s, totol time:" << op_total_time[i] << "sec";
  }
  VLOG(0) << "card: " << thread_id_ << " read time: " << read_time
          << ", percent: " << read_time / total_time * 100;
  return;
}

void PSGPUWorker::ResetStat() {
  total_time_ = 0;
  read_time_ = 0;
  pack_time_ = 0;
  pull_sparse_local_time_ = 0;
  op_all_time_ = 0;
  xpu_op_time_ = 0;
  xpu_wait_time_ = 0;
  cpu_op_time_ = 0;
  collect_label_time_ = 0;
  fill_sparse_time_ = 0;
  push_sparse_time_ = 0;
  gpu_2_cpu_time_ = 0;
  cpu_2_gpu_time_ = 0;
  total_inst_ = 0;
}

void PSGPUWorker::ProduceTasks() { return; }

}  // end namespace framework
}  // end namespace paddle
#endif
