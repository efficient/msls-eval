#include "meshdb.h"
#include "util.h"
// #include <unordered_map>

MeshDB::MeshDB(const MeshDBParams& params, Stat& stat,
               MeshDBItemLifetimeInfo* lifetime_info)
    : params_(params), stat_(stat), lifetime_info_(lifetime_info) {
  log_bytes_ = 0;
  next_version_ = 0;
  updates_since_last_compaction_ = 0;
  next_compaction_key_ = 0;
  compaction_rand_seed_ = 0;

  for (std::size_t i = 0; i < 1 + num_lifetime_classes; i++)
    level_bytes_[i] = 0;

  // for (std::size_t i = 0; i < num_lifetime_classes; i++)
  //     compaction_weight_[i] = 1. / static_cast<double>(num_lifetime_classes);
  compaction_weight_[0] = 1.;
  for (std::size_t i = 1; i < num_lifetime_classes; i++)
    compaction_weight_[i] = compaction_weight_[i - 1] / 10;
  // compaction_weight_[i] = compaction_weight_[i - 1] / 4;

  compaction_weight_[0] = 1. / 18000.;
  compaction_weight_[1] = 1. / 130000.;
  compaction_weight_[2] = 1. / 1000000.;

  global_mutation_rate_ = 0.5;
  for (std::size_t i = 0; i < num_lifetime_classes; i++)
    level_mutation_rate_[i] = 0.5;

  for (std::size_t i = 0; i < num_lifetime_classes - 1; i++)
    lifetime_threshold_[i] = static_cast<double>(i + 1);
}

MeshDB::~MeshDB() {}

void MeshDB::print_status() const {
  printf("log: %zu items, %lu bytes\n", log_.size(), log_bytes_);
  for (std::size_t i = 0; i < 1 + num_lifetime_classes; i++)
    printf("level-%zu: %zu tables, %lu bytes\n", i, levels_[i].size(),
           level_bytes_[i]);
}

void MeshDB::dump_state(FILE* fp) const {
  // XXX: Memtable is not dumped now.
  fprintf(fp, "next_version:%lu\n", next_version_);

  fprintf(fp, "log:\n");
  dump_state(fp, log_);

  fprintf(fp, "levels:\n");
  for (std::size_t level = 0; level < 1 + num_lifetime_classes; level++) {
    auto& sstables = levels_[level];
    fprintf(fp, "level:\n");
    for (std::size_t i = 0; i < sstables.size(); i++) {
      fprintf(fp, "sstable:\n");
      dump_state(fp, *sstables[i]);
    }
  }
}

void MeshDB::dump_state(FILE* fp, const sstable_t& l) {
  for (std::size_t i = 0; i < l.size(); i++) dump_state(fp, l[i]);
}

void MeshDB::dump_state(FILE* fp, const MeshDBItem& item) {
  fprintf(fp, "item:%lu,%lu,%lu,%s\n", item.key, item.version, item.size,
          item.deletion ? "T" : "F");
}

void MeshDB::put(uint64_t key, uint64_t item_size) {
  MeshDBItem item{key, next_version_++, item_size, false};
  append_to_log(item);
}

void MeshDB::del(uint64_t key) {
  MeshDBItem item{key, next_version_++, 16, true};
  append_to_log(item);
}

uint64_t MeshDB::get(MeshDBKey key) {
  // TODO: Implement
  (void)key;
  return 0;
}

void MeshDB::force_compact() {
  flush_log();

  std::size_t num_steps = 10;

  next_compaction_key_ = 0;
  while (next_compaction_key_ < 2000000) {
    // XXX: hardcoding to set a key range
    for (std::size_t step = 0; step < num_steps; step++) {
      MeshDBKey first = next_compaction_key_;
      MeshDBKey last =
          next_compaction_key_ +
          2000000 / params_.level0_sstable_count_threshold / num_steps;
      next_compaction_key_ +=
          2000000 / params_.level0_sstable_count_threshold / num_steps;

      compact(1 + num_lifetime_classes, first, last);
    }
  }
  next_compaction_key_ = 0;
}

void MeshDB::append_to_log(const MeshDBItem& item) {
  log_.push_back(item);

  // Update statistics.
  auto new_log_bytes = log_bytes_ + item.size;
  // auto log_bytes_d = log_bytes_ / 4096;
  // auto new_log_bytes_d = new_log_bytes / 4096;
  // if (log_bytes_d != new_log_bytes_d) {
  //     // New blocks are written.
  //     stat_.write((new_log_bytes_d - log_bytes_d) * 4096);
  // }
  stat_.write(item.size);
  log_bytes_ = new_log_bytes;

  updates_since_last_compaction_ += 1;

  if (log_bytes_ > params_.log_size_threshold) flush_log();
}

void MeshDB::flush_log() {
  if (log_.size() == 0) return;

  // Simplified for simulation; a new SSTable is created from the memtable,
  // causing no disk read.
  item_ptr_t items;
  item_ptr_t items2;
  sort_items(log_, items);
  deduplicate_items(items, items2);
  sstable_locs_t new_sstable_locs;
  create_sstables(1, items2, new_sstable_locs);
  delete_log();

  double accum_p[num_lifetime_classes];

  static int r = 0;
  if (r++ % 100 == 0) {
    printf("global_mutation_rate=%lf\n", global_mutation_rate_);
    for (std::size_t i = 0; i < num_lifetime_classes; i++)
      printf("level_mutation_rate[%zu]=%lf\n", i, level_mutation_rate_[i]);
    for (std::size_t i = 0; i < num_lifetime_classes; i++)
      printf("compaction_weight[%zu]=%lf\n", i, compaction_weight_[i]);
    // for (std::size_t i = 0; i < num_lifetime_classes - 1; i++)
    //     printf("lifetime_threshold[%zu]=%lf\n", i, lifetime_threshold_[i]);
  }

  // std::size_t num_steps = 10;
  std::size_t num_steps = 1;

  for (std::size_t step = 0; step < num_steps; step++) {
    double accum_weight = 0.;
    {
      std::size_t lifetime_class = num_lifetime_classes - 1;
      while (true) {
        accum_weight += compaction_weight_[lifetime_class];
        accum_p[lifetime_class] = accum_weight;
        if (lifetime_class == 0) break;
        lifetime_class--;
      }
    }

    double r = fast_rand_d(&compaction_rand_seed_) * accum_weight;

    std::size_t num_levels = 2;
    {
      std::size_t lifetime_class = num_lifetime_classes - 1;
      while (true) {
        if (r < accum_p[lifetime_class]) {
          num_levels = 2 + lifetime_class;
          break;
        }
        if (lifetime_class == 0) break;
        lifetime_class--;
      }
    }

    // XXX: hardcoding to set a key range
    MeshDBKey first = next_compaction_key_;
    MeshDBKey last =
        next_compaction_key_ +
        2000000 / (params_.level0_sstable_count_threshold + 1) / num_steps;
    next_compaction_key_ +=
        2000000 / (params_.level0_sstable_count_threshold + 1) / num_steps;
    if (next_compaction_key_ >= 2000000) next_compaction_key_ = 0;

    bool any_key_in_level0 = true;
    std::vector<std::size_t> sstable_indices;
    find_overlapping_tables(0, first, last, sstable_indices);
    for (auto i : sstable_indices)
      for (auto& item : *levels_[0][i])
        if (first <= item.key && item.key <= last) {
          any_key_in_level0 = true;
          break;
        }
    if (any_key_in_level0 /*|| num_levels > 2*/) {
      // printf("compact: num_levels=%zu first=%lu last=%lu\n", num_levels,
      // first, last);
      compact(num_levels, first, last);
    }
  }

  // printf("\n");
}

void MeshDB::delete_log() {
  // stat_.del(log_bytes_ / 4096 * 4096);
  stat_.del(log_bytes_);
  log_.clear();
  log_bytes_ = 0;
}

struct _MeshDBDereferenceComparer {
  bool operator()(const MeshDBItem* a, const MeshDBItem* b) const {
    auto& item_a = *a;
    auto& item_b = *b;
    if (item_a.key < item_b.key)
      return true;
    else if (item_a.key == item_b.key && item_a.version < item_b.version)
      return true;
    return false;
  }
};

void MeshDB::sort_items(sstable_t& items, item_ptr_t& out_items) {
  std::size_t count = items.size();
  out_items.clear();
  out_items.reserve(count);
  for (auto& item : items) out_items.push_back(&item);
  std::sort(out_items.begin(), out_items.end(), _MeshDBDereferenceComparer());
}

struct _MeshDBSSTableComparer {
  const MeshDB::sstables_t& sstables;
  const std::vector<std::size_t>& sstables_pos;

  bool operator()(const std::size_t& a, const std::size_t& b) const {
    auto& item_a = (*sstables[a])[sstables_pos[a]];
    auto& item_b = (*sstables[b])[sstables_pos[b]];
    if (item_a.key > item_b.key)
      return true;
    else if (item_a.key == item_b.key && item_a.version > item_b.version)
      return true;
    return false;
  }
};

void MeshDB::merge_items(const sstables_t& sstables, item_ptr_t& out_items) {
  std::size_t total_count = 0;
  std::vector<std::size_t> heap;
  std::vector<std::size_t> sstable_pos;
  for (std::size_t i = 0; i < sstables.size(); i++) {
    total_count += sstables[i]->size();
    if (sstables[i]->size() > 0) heap.push_back(i);
    sstable_pos.push_back(0);
  }

  out_items.clear();
  out_items.reserve(total_count);

  // Since std::make_heap makes a max-heap, we use a comparator with the
  // opposite result.
  _MeshDBSSTableComparer comp{sstables, sstable_pos};

  std::make_heap(heap.begin(), heap.end(), comp);
  while (heap.size() > 0) {
    auto sstable_index = heap.front();

    std::pop_heap(heap.begin(), heap.end(), comp);
    heap.pop_back();

    auto& sstable = sstables[sstable_index];
    // assert(sstable_pos[sstable_index] < sstable->size());
    auto& item = (*sstable)[sstable_pos[sstable_index]++];

    out_items.push_back(&item);
    // if (out_items.size() >= 2)
    //     assert(out_items[out_items.size() - 2]->key <=
    //     out_items[out_items.size() - 1]->key);

    if (sstable_pos[sstable_index] < sstable->size()) {
      heap.push_back(sstable_index);
      std::push_heap(heap.begin(), heap.end(), comp);
    }
  }

  // assert(out_items.size() == total_count);
  // for (std::size_t i = 0; i < sstables.size(); i++)
  //     assert(sstable_pos[i] == sstables[i]->size());
}

void MeshDB::deduplicate_items(const item_ptr_t& items, item_ptr_t& out_items) {
  std::size_t count = items.size();

  out_items.clear();
  if (count == 0) return;
  out_items.reserve(count);

  for (std::size_t i = 0; i < count - 1; i++) {
    if (items[i]->key != items[i + 1]->key) out_items.push_back(items[i]);
  }
  if (count > 0) out_items.push_back(items[count - 1]);
}

void MeshDB::insert_sstable(std::size_t level, sstable_t* sstable) {
  assert(sstable->size() != 0);

  // TODO: Use binary search to find the insert point.
  auto it = levels_[level].begin();
  std::size_t idx = 0;
  while (it != levels_[level].end() &&
         (*it)->front().key <= sstable->front().key) {
    ++it;
    idx++;
  }
  levels_[level].insert(it, sstable);
  // levels_[level].push_back(sstable);
}

MeshDB::sstable_t* MeshDB::remove_sstable(std::size_t level, std::size_t idx) {
  sstable_t* t = levels_[level][idx];

  for (auto j = idx; j < levels_[level].size() - 1; j++)
    levels_[level][j] = levels_[level][j + 1];
  levels_[level].pop_back();

  return t;
}

void MeshDB::create_sstables(std::size_t num_levels, const item_ptr_t& items,
                             sstable_locs_t& out_new_sstables) {
  const std::size_t last_level = 1 + num_lifetime_classes - 1;

  sstable_t* sstables[num_levels];
  // The current SSTable size in bytes.
  uint64_t sstable_sizes[num_levels];

  for (std::size_t i = 0; i < num_levels; i++) {
    sstables[i] = nullptr;
    sstable_sizes[i] = 0;
  }

  auto insert_f = [&](std::size_t level) {
    insert_sstable(level, sstables[level]);
    out_new_sstables.push_back(std::make_pair(sstables[level], level));
    level_bytes_[level] += sstable_sizes[level];
    stat_.write(sstable_sizes[level]);
    sstables[level] = nullptr;
    sstable_sizes[level] = 0;
  };

  for (auto& item : items) {
    std::size_t level = 1 + lifetime_info_->item_class(item->key);

    // uint64_t item_lifetime = lifetime_info_->item_lifetime(item->key);
    // std::size_t item_class;
    // for (item_class = 0; item_class < num_lifetime_classes - 1; item_class++)
    //     if (static_cast<double>(item_lifetime) <=
    //     lifetime_threshold_[item_class])
    //         break;
    // std::size_t level = 1 + item_class;

    // std::size_t level = num_levels - 1;

    if (level >= num_levels) level = num_levels - 1;

    // Deletion is discarded when there is no more levels.
    // TODO: this leaves lots of deletion tombstones if the item's lifetime
    // class is not the last one.
    if (item->deletion && level == last_level) continue;

    if (sstables[level]) {
      bool need_new_sstable = false;
      if (sstable_sizes[level] + item->size > params_.sstable_size_threshold) {
        // Stop adding new items if this SSTable become large in size.
        need_new_sstable = true;
      }

      if (need_new_sstable) insert_f(level);
    }

    if (!sstables[level]) sstables[level] = new sstable_t();
    sstables[level]->push_back(*item);
    sstable_sizes[level] += item->size;
  }
  for (std::size_t level = 0; level < num_levels; level++) {
    if (sstables[level]) {
      // Add any pending SSTable in construction.
      insert_f(level);
    }
  }
}

void MeshDB::find_overlapping_tables(
    std::size_t level, const MeshDBKey& first, const MeshDBKey& last,
    std::vector<std::size_t>& out_sstable_indices) {
  // assert(level >= 1);
  // assert(level < levels_.size());

  // TODO: Use binary search to reduce the search range.

  auto& level_tables = levels_[level];
  std::size_t count = level_tables.size();
  out_sstable_indices.clear();

  for (std::size_t i = 0; i < count; i++) {
    auto& sstable = *level_tables[i];
    if (!(last < sstable.front().key || sstable.back().key < first))
      out_sstable_indices.push_back(i);
  }
}

struct _MeshDBReverseInt {
  bool operator()(const std::size_t& a, const std::size_t& b) const {
    return a > b;
  }
};

// struct phash {
//     template <typename T, typename U>
//     std::size_t operator()(const std::pair<T, U>& x) const {
//         return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
//     }
// };

void MeshDB::compact(std::size_t num_levels, const MeshDBKey& first,
                     const MeshDBKey& last) {
  std::vector<sstable_t*> merge_source;
  std::vector<std::pair<std::size_t, std::size_t>> sstables_to_delete;

  // std::unordered_map<std::pair<MeshDBKey, uint64_t>, std::size_t, phash>
  // org_level;
  std::size_t org_size = 0;

  for (std::size_t level = 0; level < num_levels; level++) {
    std::vector<std::size_t> sstable_indices;
    find_overlapping_tables(level, first, last, sstable_indices);

    sstable_t* temp_sstable;
    if (level > 0) {
      temp_sstable = new sstable_t();
      merge_source.push_back(temp_sstable);
    }

    for (auto& i : sstable_indices) {
      if (level == 0) {
        temp_sstable = new sstable_t();
        merge_source.push_back(temp_sstable);
      }

      auto& org_sstable = *levels_[level][i];

      uint64_t sstable_size = 0;
      std::size_t item_start = 0;
      while (item_start < org_sstable.size() &&
             org_sstable[item_start].key < first)
        item_start++;

      std::size_t item_end = item_start;
      while (item_end < org_sstable.size() &&
             org_sstable[item_end].key <= last) {
        temp_sstable->push_back(org_sstable[item_end]);
        if (level == num_levels - 1) {
          // org_level[std::make_pair(org_sstable[item_end].key,
          // org_sstable[item_end].version)] = level;
          org_size += org_sstable[item_end].size;
        }
        sstable_size += org_sstable[item_end].size;
        item_end++;
      }

      stat_.read(sstable_size);
      stat_.del(sstable_size);
      level_bytes_[level] -= sstable_size;

      org_sstable.erase(
          org_sstable.begin() + static_cast<std::ptrdiff_t>(item_start),
          org_sstable.begin() + static_cast<std::ptrdiff_t>(item_end));
      if (org_sstable.size() == 0) {
        sstables_to_delete.push_back(std::make_pair(level, i));
        delete &org_sstable;
      }
    }
  }

  std::reverse(sstables_to_delete.begin(), sstables_to_delete.end());
  for (auto p : sstables_to_delete) remove_sstable(p.first, p.second);

  item_ptr_t items;
  merge_items(merge_source, items);

  item_ptr_t items2;
  deduplicate_items(items, items2);

  // Calculate the mutation rate and modify the compaction weight
  std::size_t unmodified_size = 0;
  if (org_size != 0) {
    for (auto& item : items2) {
      if (item - &merge_source.back()->front() >= 0 &&
          &merge_source.back()->back() - item >= 0)
        unmodified_size += item->size;
    }
  }

  // Create new SSTables.
  sstable_locs_t new_sstable_locs;
  create_sstables(num_levels, items2, new_sstable_locs);

  // Delete old SSTables.
  for (auto& sstable : merge_source) delete sstable;

  // Calculate the mutation rate and modify the compaction weight
  // std::size_t unmodified_size = 0;
  // for (auto& p : new_sstable_locs) {
  //     if (p.second != num_levels - 1)
  //         continue;
  //     for (auto& item : *p.first) {
  //         if (org_level[std::make_pair(item.key, item.version)] == p.second)
  //             unmodified_size += item.size;
  //     }
  // }
  double mutation_rate;
  if (org_size != 0)
    mutation_rate =
        1. -
        static_cast<double>(unmodified_size) / static_cast<double>(org_size);
  else
    mutation_rate = 0.;
  if (mutation_rate < 0.) mutation_rate = 0.;
  if (mutation_rate > 1.) mutation_rate = 1.;

  // printf("num_levels=%zu mutation_rate=%lf\n", num_levels, mutation_rate);
  // if (mutation_rate < params_.target_mutation_rate) {
  //     // Too little mutation; decrease weight for less frequent compaction.
  //     compaction_weight_[num_levels - 2] /= 1.01;
  // }
  // else {
  //     // Too much mutation; increase weight for more frequent compaction.
  //     compaction_weight_[num_levels - 2] *= 1.01;
  // }

  // double weight_sum = 0.;
  // for (std::size_t i = 0; i < num_lifetime_classes; i++)
  //     weight_sum += compaction_weight_[i];

  // if (num_levels < 1 + num_lifetime_classes) {
  //     // if (mutation_rate * compaction_weight_[num_levels - 2] <
  //     level_mutation_rate_[num_levels - 1] * compaction_weight_[num_levels -
  //     1])
  //     if (mutation_rate < level_mutation_rate_[num_levels - 1])
  //         // compaction_weight_[num_levels - 2] /= pow(1.01, (1. /
  //         (compaction_weight_[num_levels - 2] / weight_sum)));
  //         compaction_weight_[num_levels - 2] /= 1.01;
  //     else
  //         // compaction_weight_[num_levels - 2] *= pow(1.01, (1. /
  //         (compaction_weight_[num_levels - 2] / weight_sum)));
  //         compaction_weight_[num_levels - 2] *= 1.01;
  // }

  // Normalize weights
  double weight_sum = 0.;
  for (std::size_t i = 0; i < num_lifetime_classes; i++)
    weight_sum += compaction_weight_[i];
  for (std::size_t i = 0; i < num_lifetime_classes; i++)
    compaction_weight_[i] /= weight_sum;

  global_mutation_rate_ = global_mutation_rate_ * 0.99 + mutation_rate * 0.01;
  level_mutation_rate_[num_levels - 2] =
      level_mutation_rate_[num_levels - 2] * 0.99 + mutation_rate * 0.01;
}
