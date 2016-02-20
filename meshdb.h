#pragma once

#include "common.h"
#include "stat.h"
#include <cstdio>
#include <unordered_map>

typedef uint64_t MeshDBKey;

struct MeshDBParams {
  // When a log file exceeds this size, a new Level-0 SSTable is created, and a
  // new log file is created.
  uint64_t log_size_threshold;
  // When the level 0 ("young") has this many SSTables, all of them are merged
  // into the next level.
  uint64_t level0_sstable_count_threshold;
  // When an SSTable file exceeds this size, a new SSTable is created.
  uint64_t sstable_size_threshold;
  // Adjust compaction frequency to meet this mutation rate for SSTables in the
  // last level of compaction.
  double target_mutation_rate;

  MeshDBParams() {
    log_size_threshold = 4 * 1048576;
    level0_sstable_count_threshold = 4;
    // level0_sstable_count_threshold = 12;
    sstable_size_threshold = 2 * 1048576;
    target_mutation_rate = 0.10;
  }
};

struct MeshDBItem {
  MeshDBKey key;
  uint64_t version;
  uint64_t size;
  bool deletion;
};

class MeshDBItemLifetimeInfo {
 public:
  virtual ~MeshDBItemLifetimeInfo() {}
  virtual std::size_t item_class(MeshDBKey key) {
    (void)key;
    return 0;
  }
  virtual uint64_t item_lifetime(MeshDBKey key) {
    (void)key;
    return 1;
  }
  virtual uint64_t class_lifetime(std::size_t lifetime_class) {
    (void)lifetime_class;
    return 1;
  }
};

// A MeshDB
class MeshDB {
 public:
  static const std::size_t num_lifetime_classes =
      3;  // TODO: Allow custom class count.
  // static const std::size_t num_lifetime_classes = 5;      // TODO: Allow
  // custom class count.

  MeshDB(const MeshDBParams& params, Stat& stat,
         MeshDBItemLifetimeInfo* lifetime_info);
  ~MeshDB();

  // Prints the summary of the store.
  void print_status() const;

  // Writes the current items in the store to the file.
  void dump_state(FILE* fp) const;

  // Puts a new item in the store.
  void put(MeshDBKey key, uint64_t item_size);

  // Deletes an item from the store.
  void del(MeshDBKey key);

  // Gets an item from the store.
  uint64_t get(MeshDBKey key);

  // Forces compaction until there is no successor SSTable.
  void force_compact();

  typedef std::vector<MeshDBItem> sstable_t;
  typedef std::vector<sstable_t*> sstables_t;
  typedef std::vector<std::pair<sstable_t*, std::size_t>> sstable_locs_t;

  typedef std::vector<MeshDBItem*> item_ptr_t;

 protected:
  // Adds a new item to the log.
  void append_to_log(const MeshDBItem& item);

  // Flushes all in-memory data to disk.  This effectively creates new level-0
  // SSTables from the Memtable.
  void flush_log();

  // Deletes the log.
  void delete_log();

  // Sorts items.
  void sort_items(sstable_t& items, item_ptr_t& out_items);

  // Merges SSTables.
  void merge_items(const sstables_t& sstables, item_ptr_t& out_items);

  // Removes duplicate items and takes the latter ones.  The items must be
  // sorted by key.
  void deduplicate_items(const item_ptr_t& items, item_ptr_t& out_items);

  // Creates new SSTables from the items designated by the indices.
  void create_sstables(std::size_t num_levels, const item_ptr_t& items,
                       sstable_locs_t& out_new_sstables);

  // Finds all overlapping SSTables in the level.
  void find_overlapping_tables(std::size_t level, const MeshDBKey& first,
                               const MeshDBKey& last,
                               std::vector<std::size_t>& out_sstable_indices);

  // Performs compaction with SSTables from the level and all over overlapping
  // SSTables in the next level.
  void compact(std::size_t num_levels, const MeshDBKey& first,
               const MeshDBKey& last);

  // Inserts a new SSTable into the level.
  void insert_sstable(std::size_t level, sstable_t* sstable);

  // Removes an SSTable from the level.  This does not release the memory used
  // by the SSTable.
  sstable_t* remove_sstable(std::size_t level, std::size_t idx);

  // Writes an item list to the file.
  static void dump_state(FILE* fp, const sstable_t& l);
  static void dump_state(FILE* fp, const MeshDBItem& item);

 private:
  MeshDBParams params_;
  Stat& stat_;
  MeshDBItemLifetimeInfo* lifetime_info_;
  sstable_t log_;
  uint64_t log_bytes_;
  sstables_t levels_[1 + num_lifetime_classes];
  uint64_t level_bytes_[1 + num_lifetime_classes];
  uint64_t next_version_;
  uint64_t updates_since_last_compaction_;
  MeshDBKey next_compaction_key_;
  uint64_t compaction_rand_seed_;
  double compaction_weight_[num_lifetime_classes];
  double global_mutation_rate_;
  double level_mutation_rate_[num_lifetime_classes];
  double lifetime_threshold_[num_lifetime_classes - 1];
};
