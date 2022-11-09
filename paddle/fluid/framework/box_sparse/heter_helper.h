#pragma once
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"

template <typename KeyType, typename ValType, typename GradType>
class HeterHelper {
public:
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len,
                size_t chunk_size, int stream_num);
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len);
  void end_pass();
protected:
  using Table = HashTable<KeyType, ValType>;
  std::vector<Table*> tables_;
}