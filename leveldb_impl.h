#pragma once

#include "leveldb.h"
#include <pthread.h>

namespace leveldb {
// For forward declaration.
class DB;
class Options;
}

// An interface to the LevelDB implementation
class LevelDBImpl {
  friend class LevelDBSequentialFile;
  friend class LevelDBRandomAccessFile;
  friend class LevelDBWritableFile;
  friend class LevelDBEnv;

 public:
  LevelDBImpl(const LevelDBParams& params, std::vector<Stat>& stats);
  ~LevelDBImpl();

  // Prints the summary of the store.
  void print_status() const;

  // Writes the current items in the store to the file.
  void dump_state(FILE* fp) const;

  // Puts a new item in the store.
  void put(LevelDBKey key, uint32_t item_size);

  // Deletes an item from the store.
  void del(LevelDBKey key);

  // Gets an item from the store.
  uint64_t get(LevelDBKey key);

  // Forces compaction until there is no SSTable except the last level.
  void force_compact();

 protected:
  void Read(std::size_t len);
  void Append(std::size_t len);
  void Delete(std::size_t len);

 private:
  LevelDBParams params_;
  std::vector<Stat>& stats_;

  leveldb::Options* options_;
  leveldb::DB* db_;

  pthread_mutex_t stats_mutex_;
  volatile uint64_t read_;
  volatile uint64_t appended_;

  char value_buf_[1024];
};
