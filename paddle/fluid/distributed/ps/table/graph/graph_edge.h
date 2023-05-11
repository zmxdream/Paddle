// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
namespace paddle {
namespace distributed {

class GraphEdgeBlob {
 public:
  GraphEdgeBlob() {}
  virtual ~GraphEdgeBlob() {}
  size_t size() { return id_arr.size(); }
  virtual void add_edge(int64_t id, float weight);
  int64_t get_id(int idx) { return id_arr[idx]; }
  virtual half get_weight(int idx) { return (half)(1.0); }
  std::vector<int64_t>& export_id_array() { return id_arr; }

  // ==== adapt for edge feature === 
  virtual int get_feature_ids(int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    return 0;
  }
  virtual int get_float_feature(int slot_idx,
                              std::vector<float> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    return 0;
  }
  virtual void set_feature(int idx, const std::string &str) {}
  virtual void set_feature_size(int size) {}
  virtual void shrink_to_fit() {}
  virtual int get_feature_size() { return 0; }
  // ==== adapt for edge feature === 

 protected:
  std::vector<int64_t> id_arr;
};

class WeightedGraphEdgeBlob : public GraphEdgeBlob {
 public:
  WeightedGraphEdgeBlob() {}
  virtual ~WeightedGraphEdgeBlob() {}
  virtual void add_edge(int64_t id, float weight);
  virtual half get_weight(int idx) { return weight_arr[idx]; }

 protected:
  std::vector<half> weight_arr;
};


class GraphEdgeBlobWithFeature : public GraphEdgeBlob {
 public:
  GraphEdgeBlobWithFeature() {}
  virtual ~GraphEdgeBlobWithFeature() {}
  virtual void add_edge(int64_t id, float weight);
  virtual half get_weight(int idx) { return weight_arr[idx]; }

 // === to adapt ====
 virtual int get_feature_ids(int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (slot_idx < static_cast<int>(offset)) {
      const std::string &s = this->feature[slot_idx];
      const uint64_t *feas = (const uint64_t *)(s.c_str());
      num = s.length() / sizeof(uint64_t);
      CHECK((s.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << s << "]";
      for (size_t i = 0; i < num; ++i) {
        feature_id.push_back(feas[i]);
        slot_id.push_back(slot_idx);
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return num;
  }

  virtual int get_float_feature(int slot_idx,
                                std::vector<float> &float_feature,      // NOLINT
                                std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (offset + slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[offset + slot_idx];
      const float *feas = (const float *)(s.c_str());
      num = s.length() / sizeof(float);
      CHECK((s.length() % sizeof(float)) == 0)
          << "bad feature_item: [" << s << "]";
      for (size_t i = 0; i < num; ++i) {
        float_feature.push_back(feas[i]);
        slot_id.push_back(slot_idx);
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
             "get_feature_ids get errno should be 0, but got %d.", errno));
    return num;
  }

  virtual std::string *mutable_feature(int idx) {
    if (idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(idx + 1);
    }
    return &(this->feature[idx]);
  }

  virtual std::string *mutable_float_feature(int idx) {
    if (offset + idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(offset + idx + 1);
    }
    return &(this->feature[offset + idx]);
  }

  virtual void set_feature(int idx, const std::string &str) {
    if (idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(idx + 1);
    }
    this->feature[idx] = str;
  }
  virtual void set_feature_size(int size) { 
    this->feature.resize(size);
    offset = size;
  }
  virtual void set_float_feature_size(int size) { this->feature.resize(offset + size); }
  virtual int get_feature_size() { 
    return offset;
  }
  virtual int get_float_feature_size() { 
    return this->feature.size() - offset;
  }
  virtual void shrink_to_fit() {
    feature.shrink_to_fit();
    for (auto &slot : feature) {
      slot.shrink_to_fit();
    }
  }

  template <typename T>
  static std::string parse_value_to_bytes(std::vector<std::string> feat_str) {
    T v;
    size_t Tsize = sizeof(T) * feat_str.size();
    char buffer[Tsize];
    for (size_t i = 0; i < feat_str.size(); i++) {
      std::stringstream ss(feat_str[i]);
      ss >> v;
      std::memcpy(
          buffer + sizeof(T) * i, reinterpret_cast<char *>(&v), sizeof(T));
    }
    return std::string(buffer, Tsize);
  }

  template <typename T>
  static void parse_value_to_bytes(
      std::vector<std::string>::iterator feat_str_begin,
      std::vector<std::string>::iterator feat_str_end,
      std::string *output) {
    T v;
    size_t feat_str_size = feat_str_end - feat_str_begin;
    size_t Tsize = sizeof(T) * feat_str_size;
    char buffer[Tsize] = {'\0'};
    for (size_t i = 0; i < feat_str_size; i++) {
      std::stringstream ss(*(feat_str_begin + i));
      ss >> v;
      std::memcpy(
          buffer + sizeof(T) * i, reinterpret_cast<char *>(&v), sizeof(T));
    }
    output->assign(buffer);
  }

  template <typename T>
  static std::vector<T> parse_bytes_to_array(std::string feat_str) {
    T v;
    std::vector<T> out;
    size_t start = 0;
    const char *buffer = feat_str.data();
    while (start < feat_str.size()) {
      std::memcpy(reinterpret_cast<char *>(&v), buffer + start, sizeof(T));
      start += sizeof(T);
      out.push_back(v);
    }
    return out;
  }

  template <typename T>
  static int parse_value_to_bytes(
      std::vector<paddle::string::str_ptr>::iterator feat_str_begin,
      std::vector<paddle::string::str_ptr>::iterator feat_str_end,
      std::string *output) {
    size_t feat_str_size = feat_str_end - feat_str_begin;
    size_t Tsize = sizeof(T) * feat_str_size;
    size_t num = output->length();
    output->resize(num + Tsize);

    T *fea_ptrs = reinterpret_cast<T *>(&(*output)[num]);

    thread_local paddle::string::str_ptr_stream ss;
    for (size_t i = 0; i < feat_str_size; i++) {
      ss.reset(*(feat_str_begin + i));
      int len = ss.end - ss.ptr;
      char *old_ptr = ss.ptr;
      ss >> fea_ptrs[i];
      if (ss.ptr - old_ptr != len) {
        return -1;
      }
    }
    return 0;
  }

 // === to adapt ==== 



 protected:
  std::vector<half> weight_arr;
  std::vector<std::vector<std::string>> feature;
};





}  // namespace distributed
}  // namespace paddle
