#include "rocksdb_impl.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Winline"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#pragma GCC diagnostic pop
#include <stdlib.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>

#define OVERRIDE override
// #define OVERRIDE

// A wrapper for SequentialFile that forwards the data read information to
// RocksDBImpl.
class RocksDBSequentialFile : public rocksdb::SequentialFile {
 public:
  RocksDBSequentialFile(RocksDBImpl* rocksdb_impl,
                        std::unique_ptr<rocksdb::SequentialFile>* t)
      : rocksdb::SequentialFile(), rocksdb_impl_(rocksdb_impl), target_(t) {}

  virtual ~RocksDBSequentialFile() OVERRIDE { delete target_; }

  virtual rocksdb::Status Read(size_t n, rocksdb::Slice* result,
                               char* scratch) OVERRIDE {
    rocksdb_impl_->Read(n);
    return (*target_)->Read(n, result, scratch);
  }

  virtual rocksdb::Status Skip(uint64_t n) OVERRIDE {
    return (*target_)->Skip(n);
  }

  virtual rocksdb::Status InvalidateCache(size_t offset,
                                          size_t length) OVERRIDE {
    return (*target_)->InvalidateCache(offset, length);
  }

 private:
  class RocksDBImpl* rocksdb_impl_;
  std::unique_ptr<rocksdb::SequentialFile>* target_;
};

// A wrapper for RandomAccessFile that forwards the data read information to
// RocksDBImpl.
class RocksDBRandomAccessFile : public rocksdb::RandomAccessFile {
 public:
  RocksDBRandomAccessFile(RocksDBImpl* rocksdb_impl,
                          std::unique_ptr<rocksdb::RandomAccessFile>* t)
      : rocksdb::RandomAccessFile(), rocksdb_impl_(rocksdb_impl), target_(t) {}

  virtual ~RocksDBRandomAccessFile() OVERRIDE { delete target_; }

  virtual rocksdb::Status Read(uint64_t offset, size_t n,
                               rocksdb::Slice* result,
                               char* scratch) const OVERRIDE {
    rocksdb_impl_->Read(n);
    return (*target_)->Read(offset, n, result, scratch);
  }

  virtual size_t GetUniqueId(char* id, size_t max_size) const OVERRIDE {
    return (*target_)->GetUniqueId(id, max_size);
  }

  virtual void Hint(AccessPattern pattern) OVERRIDE {
    (*target_)->Hint(pattern);
  }

  virtual rocksdb::Status InvalidateCache(size_t offset,
                                          size_t length) OVERRIDE {
    return (*target_)->InvalidateCache(offset, length);
  }

 private:
  class RocksDBImpl* rocksdb_impl_;
  std::unique_ptr<rocksdb::RandomAccessFile>* target_;
};

// A wrapper for WritableFile that forwards the data append information to
// RocksDBImpl.
class RocksDBWritableFile : public rocksdb::WritableFile {
 public:
  RocksDBWritableFile(RocksDBImpl* rocksdb_impl,
                      std::unique_ptr<rocksdb::WritableFile>* t)
      : rocksdb::WritableFile(), rocksdb_impl_(rocksdb_impl), target_(t) {}

  virtual ~RocksDBWritableFile() OVERRIDE { delete target_; }

  virtual rocksdb::Status Append(const rocksdb::Slice& data) OVERRIDE {
    rocksdb_impl_->Append(data.size());
    return (*target_)->Append(data);
  }

  virtual rocksdb::Status Close() OVERRIDE { return (*target_)->Close(); }

  virtual rocksdb::Status Flush() OVERRIDE { return (*target_)->Flush(); }

  virtual rocksdb::Status Sync() OVERRIDE {
    if (rocksdb_impl_->params_.enable_fsync)
      return (*target_)->Sync();
    else {
      // Let's ignore Sync() for faster experiments.
      return rocksdb::Status::OK();
    }
  }

  virtual rocksdb::Status Fsync() OVERRIDE {
    if (rocksdb_impl_->params_.enable_fsync)
      return (*target_)->Fsync();
    else {
      // Let's ignore Fsync() for faster experiments.
      return rocksdb::Status::OK();
    }
  }

  virtual bool IsSyncThreadSafe() const OVERRIDE {
    return (*target_)->IsSyncThreadSafe();
  }

  virtual void SetIOPriority(rocksdb::Env::IOPriority pri) OVERRIDE {
    (*target_)->SetIOPriority(pri);
  }

  virtual rocksdb::Env::IOPriority GetIOPriority() OVERRIDE {
    return (*target_)->GetIOPriority();
  }

  virtual uint64_t GetFileSize() OVERRIDE { return (*target_)->GetFileSize(); }

  virtual void GetPreallocationStatus(size_t* block_size,
                                      size_t* last_allocated_block) OVERRIDE {
    (*target_)->GetPreallocationStatus(block_size, last_allocated_block);
  }

  virtual size_t GetUniqueId(char* id, size_t max_size) const OVERRIDE {
    return (*target_)->GetUniqueId(id, max_size);
  }

  virtual rocksdb::Status InvalidateCache(size_t offset,
                                          size_t length) OVERRIDE {
    return (*target_)->InvalidateCache(offset, length);
  }

 private:
  class RocksDBImpl* rocksdb_impl_;
  std::unique_ptr<rocksdb::WritableFile>* target_;
};

class RocksDBDirectory : public rocksdb::Directory {
 public:
  RocksDBDirectory(RocksDBImpl* rocksdb_impl,
                   std::unique_ptr<rocksdb::Directory>* t)
      : rocksdb::Directory(), rocksdb_impl_(rocksdb_impl), target_(t) {}

  virtual ~RocksDBDirectory() OVERRIDE { delete target_; }

  virtual rocksdb::Status Fsync() OVERRIDE {
    if (rocksdb_impl_->params_.enable_fsync)
      return (*target_)->Fsync();
    else {
      // Let's ignore Fsync() for faster experiments.
      return rocksdb::Status::OK();
    }
  }

 private:
  class RocksDBImpl* rocksdb_impl_;
  std::unique_ptr<rocksdb::Directory>* target_;
};

// A wrapper for Env that forwards the file deletion information to RocksDBImpl.
class RocksDBEnv : public rocksdb::EnvWrapper {
 public:
  RocksDBEnv(RocksDBImpl* rocksdb_impl)
      : rocksdb::EnvWrapper(rocksdb::Env::Default()),
        rocksdb_impl_(rocksdb_impl) {}

  virtual ~RocksDBEnv() OVERRIDE {}

  virtual rocksdb::Status NewSequentialFile(
      const std::string& f, std::unique_ptr<rocksdb::SequentialFile>* r,
      const rocksdb::EnvOptions& options) OVERRIDE {
    std::unique_ptr<rocksdb::SequentialFile>* r2 =
        new std::unique_ptr<rocksdb::SequentialFile>();
    rocksdb::Status status = target()->NewSequentialFile(f, r2, options);
    if (*r2 != NULL)
      r->reset(new RocksDBSequentialFile(rocksdb_impl_, r2));
    else
      delete r2;
    return status;
  }

  virtual rocksdb::Status NewRandomAccessFile(
      const std::string& f, std::unique_ptr<rocksdb::RandomAccessFile>* r,
      const rocksdb::EnvOptions& options) OVERRIDE {
    std::unique_ptr<rocksdb::RandomAccessFile>* r2 =
        new std::unique_ptr<rocksdb::RandomAccessFile>();
    rocksdb::Status status = target()->NewRandomAccessFile(f, r2, options);
    if (*r2 != NULL)
      r->reset(new RocksDBRandomAccessFile(rocksdb_impl_, r2));
    else
      delete r2;
    return status;
  }

  virtual rocksdb::Status NewWritableFile(
      const std::string& f, std::unique_ptr<rocksdb::WritableFile>* r,
      const rocksdb::EnvOptions& options) OVERRIDE {
    std::unique_ptr<rocksdb::WritableFile>* r2 =
        new std::unique_ptr<rocksdb::WritableFile>();
    rocksdb::Status status = target()->NewWritableFile(f, r2, options);
    if (*r2 != NULL)
      r->reset(new RocksDBWritableFile(rocksdb_impl_, r2));
    else
      delete r2;
    return status;
  }

  virtual rocksdb::Status NewDirectory(
      const std::string& f, std::unique_ptr<rocksdb::Directory>* r) OVERRIDE {
    std::unique_ptr<rocksdb::Directory>* r2 =
        new std::unique_ptr<rocksdb::Directory>();
    rocksdb::Status status = target()->NewDirectory(f, r2);
    if (*r2 != NULL)
      r->reset(new RocksDBDirectory(rocksdb_impl_, r2));
    else
      delete r2;
    return status;
  }

  virtual rocksdb::Status DeleteFile(const std::string& f) OVERRIDE {
    struct stat st;
    memset(&st, 0, sizeof(st));
    // XXX: The file length *might* not be as large as its actual content
    // because the directory metadata can be updated later than the appends.
    int ret = stat(f.c_str(), &st);
    if (ret == 0) rocksdb_impl_->Delete(static_cast<uint64_t>(st.st_size));

    return target()->DeleteFile(f);
  }

 private:
  class RocksDBImpl* rocksdb_impl_;
};

RocksDBImpl::RocksDBImpl(const LevelDBParams& params, std::vector<Stat>& stats)
    : params_(params), stats_(stats) {
  stats_.push_back(Stat());

  pthread_mutex_init(&stats_mutex_, NULL);
  read_ = 0;
  appended_ = 0;

  // Clean up old files.
  rocksdb::DestroyDB("rocksdb_files", rocksdb::Options());

  options_ = new rocksdb::Options();

  options_->create_if_missing = true;

  // Turn off Snappy.
  options_->compression = rocksdb::CompressionType::kNoCompression;

  // Use our Env to gather statistics.
  options_->env = new RocksDBEnv(this);

  // Limit the max open file count.
  options_->max_open_files = 900;

  // Configure the write buffer size.
  options_->write_buffer_size = params.log_size_threshold;

  // Do not overload insert.
  options_->level0_file_num_compaction_trigger = 4;
  options_->level0_slowdown_writes_trigger = 4;
  options_->level0_stop_writes_trigger = 4;

  // Use LevelDB-style table selection.
  if (params_.compaction_mode == LevelDBCompactionMode::kRocksDBMaxSize ||
      params_.compaction_mode == LevelDBCompactionMode::kRocksDBMaxSizeMT)
    options_->use_leveldb_table_selection = false;
  else if (params_.compaction_mode == LevelDBCompactionMode::kRocksDBLinear ||
           params_.compaction_mode == LevelDBCompactionMode::kRocksDBLinearMT)
    options_->use_leveldb_table_selection = true;
  else if (params_.compaction_mode ==
           LevelDBCompactionMode::kRocksDBUniversal) {
    options_->use_leveldb_table_selection =
        false;  // This will be ignored anyway.
    options_->compaction_style = rocksdb::kCompactionStyleUniversal;
    // Use a bit more level-0 files
    // options_->level0_file_num_compaction_trigger = 8;
    options_->level0_file_num_compaction_trigger = 12;
    // We have to adjust the maximum level-0 file count because RocksDB is stuck
    // with a deadlock otherwise.
    options_->level0_slowdown_writes_trigger =
        options_->level0_file_num_compaction_trigger + 2;
    options_->level0_stop_writes_trigger =
        options_->level0_file_num_compaction_trigger + 2;

    // Adjust size_ratio to handle skewed workloads gracefully without having to
    // increase the file count much.
    // options.compaction_options_universal.size_ratio = 10;
  } else
    assert(false);

  // Use multiple threads if requested.
  if (params_.compaction_mode == LevelDBCompactionMode::kRocksDBMaxSizeMT ||
      params_.compaction_mode == LevelDBCompactionMode::kRocksDBLinearMT) {
    // 1 thread is dedicated as a "background flush" thread
    // (DBOptions::IncreaseParallelism() in rocksdb/util/options.cc)
    options_->IncreaseParallelism(4 + 1);
  }

  // Turn off checksumming for faster experiments (even though we already
  // disabled crc32c).
  // options_->verify_checksums_in_compaction = false;

  // Use custom level sizes
  if (params_.use_custom_sizes) {
    std::size_t* custom_level_sizes = new std::size_t[20];

    std::ifstream ifs("output_sensitivity.txt");
    while (!ifs.eof()) {
      std::string line;
      std::getline(ifs, line);

      std::istringstream iss(line);
      std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                      std::istream_iterator<std::string>{}};

      if (tokens.size() < 5) continue;
      if (tokens[0] != "sensitivity_item_count_leveldb_best_sizes" &&
          tokens[0] != "sensitivity_log_size_leveldb_best_sizes")
        continue;
      if (static_cast<uint64_t>(atol(tokens[1].c_str())) !=
          params_.hint_num_unique_keys)
        continue;
      if (atof(tokens[2].c_str()) != params_.hint_theta) continue;
      if (static_cast<uint64_t>(atol(tokens[3].c_str())) !=
          params_.log_size_threshold)
        continue;

      options_->custom_level_size_count = tokens.size() - 5 + 1;

      custom_level_sizes[0] = 0;
      std::size_t level;
      for (level = 1; level < options_->custom_level_size_count; level++) {
        custom_level_sizes[level] = static_cast<size_t>(
            atof(tokens[5 + level - 1].c_str()) * 1000. + 0.5);
        printf("level-%zu: %zu\n", level, custom_level_sizes[level]);
      }
      // Make the last level very large and not spill.
      level--;
      custom_level_sizes[level] = 1000000000000000LU;
      printf("level-%zu: %zu (expanded)\n", level, custom_level_sizes[level]);
      printf("\n");
      break;
    }
    assert(options_->custom_level_size_count != 0);

    options_->custom_level_sizes = custom_level_sizes;
  }

  rocksdb::Status status = rocksdb::DB::Open(*options_, "rocksdb_files", &db_);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }

  memset(value_buf_, 0, sizeof(value_buf_));
}

RocksDBImpl::~RocksDBImpl() {
  delete db_;

  delete options_->env;
  if (params_.use_custom_sizes) delete[] options_->custom_level_sizes;
  delete options_;

  pthread_mutex_destroy(&stats_mutex_);
}

void RocksDBImpl::print_status() const {
  // Force updating stats.
  const_cast<RocksDBImpl*>(this)->Delete(0);
}

void RocksDBImpl::dump_state(FILE* fp) const {
  // TODO: Implement.
  (void)fp;
}

void RocksDBImpl::put(LevelDBKey key, uint32_t item_size) {
  // LevelDB includes the full SSTable file size during calculating the level
  // size;
  // we consider the average space overhead per item in LevelDB so that the
  // average stored size becomes similar to item_size.
  const uint32_t overhead = 18;

  rocksdb::Slice s_key(reinterpret_cast<const char*>(&key), sizeof(key));
  uint32_t value_size =
      static_cast<uint32_t>(static_cast<std::size_t>(item_size) - sizeof(key)) -
      overhead;
  assert(value_size < sizeof(value_buf_));
  rocksdb::Slice s_value(value_buf_, value_size);

  rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), s_key, s_value);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }
}

void RocksDBImpl::del(LevelDBKey key) {
  rocksdb::Slice s_key(reinterpret_cast<const char*>(&key), sizeof(key));

  rocksdb::Status status = db_->Delete(rocksdb::WriteOptions(), s_key);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }
}

uint64_t RocksDBImpl::get(LevelDBKey key) {
  rocksdb::Slice s_key(reinterpret_cast<const char*>(&key), sizeof(key));
  std::string s_value;
  uint64_t value;

  rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), s_key, &s_value);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }
  assert(s_value.size() >= sizeof(uint64_t));
  value = *reinterpret_cast<const uint64_t*>(s_value.data());
  return value;
}

void RocksDBImpl::force_compact() {
  rocksdb::CompactRangeOptions options;
  options.change_level = false;
  options.target_level = -1;
  options.target_path_id = 0;

  db_->CompactRange(options, NULL, NULL);

  // Force stat update.
  Delete(0);
}

void RocksDBImpl::Read(std::size_t len) { __sync_fetch_and_add(&read_, len); }

void RocksDBImpl::Append(std::size_t len) {
  __sync_fetch_and_add(&appended_, len);
}

void RocksDBImpl::Delete(std::size_t len) {
  uint64_t read = read_;
  __sync_fetch_and_sub(&read_, read);
  uint64_t appended = appended_;
  __sync_fetch_and_sub(&appended_, appended);

  pthread_mutex_lock(&stats_mutex_);
  if (read != 0) stats_.back().read(read);
  if (appended != 0) stats_.back().write(appended);
  if (len != 0) stats_.back().del(len);
  pthread_mutex_unlock(&stats_mutex_);
}
