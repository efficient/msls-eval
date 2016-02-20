#include "leveldb_impl.h"
#include "leveldb/db.h"
#include "leveldb/env.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>

#define OVERRIDE override
// #define OVERRIDE

// A wrapper for SequentialFile that forwards the data read information to
// LevelDBImpl.
class LevelDBSequentialFile : public leveldb::SequentialFile {
 public:
  LevelDBSequentialFile(LevelDBImpl* leveldb_impl, leveldb::SequentialFile* t)
      : leveldb::SequentialFile(), leveldb_impl_(leveldb_impl), target_(t) {}

  virtual ~LevelDBSequentialFile() OVERRIDE { delete target_; }

  virtual leveldb::Status Read(size_t n, leveldb::Slice* result,
                               char* scratch) OVERRIDE {
    leveldb_impl_->Read(n);
    return target_->Read(n, result, scratch);
  }

  virtual leveldb::Status Skip(uint64_t n) OVERRIDE { return target_->Skip(n); }

 private:
  class LevelDBImpl* leveldb_impl_;
  leveldb::SequentialFile* target_;
};

// A wrapper for RandomAccessFile that forwards the data read information to
// LevelDBImpl.
class LevelDBRandomAccessFile : public leveldb::RandomAccessFile {
 public:
  LevelDBRandomAccessFile(LevelDBImpl* leveldb_impl,
                          leveldb::RandomAccessFile* t)
      : leveldb::RandomAccessFile(), leveldb_impl_(leveldb_impl), target_(t) {}

  virtual ~LevelDBRandomAccessFile() OVERRIDE { delete target_; }

  virtual leveldb::Status Read(uint64_t offset, size_t n,
                               leveldb::Slice* result,
                               char* scratch) const OVERRIDE {
    leveldb_impl_->Read(n);
    return target_->Read(offset, n, result, scratch);
  }

 private:
  class LevelDBImpl* leveldb_impl_;
  leveldb::RandomAccessFile* target_;
};

// A wrapper for WritableFile that forwards the data append information to
// LevelDBImpl.
class LevelDBWritableFile : public leveldb::WritableFile {
 public:
  LevelDBWritableFile(LevelDBImpl* leveldb_impl, leveldb::WritableFile* t)
      : leveldb::WritableFile(), leveldb_impl_(leveldb_impl), target_(t) {}

  virtual ~LevelDBWritableFile() OVERRIDE { delete target_; }

  virtual leveldb::Status Append(const leveldb::Slice& data) OVERRIDE {
    leveldb_impl_->Append(data.size());
    return target_->Append(data);
  }

  virtual leveldb::Status Close() OVERRIDE { return target_->Close(); }

  virtual leveldb::Status Flush() OVERRIDE { return target_->Flush(); }

  virtual leveldb::Status Sync() OVERRIDE {
    if (leveldb_impl_->params_.enable_fsync)
      return target_->Sync();
    else {
      // Let's ignore Sync() for faster experiments.
      return leveldb::Status::OK();
    }
  }

 private:
  class LevelDBImpl* leveldb_impl_;
  leveldb::WritableFile* target_;
};

// A wrapper for Env that forwards the file deletion information to LevelDBImpl.
class LevelDBEnv : public leveldb::EnvWrapper {
 public:
  LevelDBEnv(LevelDBImpl* leveldb_impl)
      : leveldb::EnvWrapper(leveldb::Env::Default()),
        leveldb_impl_(leveldb_impl) {}

  virtual ~LevelDBEnv() OVERRIDE {}

  virtual leveldb::Status NewSequentialFile(
      const std::string& f, leveldb::SequentialFile** r) OVERRIDE {
    leveldb::Status status = target()->NewSequentialFile(f, r);
    if (*r != NULL) *r = new LevelDBSequentialFile(leveldb_impl_, *r);
    return status;
  }

  virtual leveldb::Status NewRandomAccessFile(
      const std::string& f, leveldb::RandomAccessFile** r) OVERRIDE {
    leveldb::Status status = target()->NewRandomAccessFile(f, r);
    if (*r != NULL) *r = new LevelDBRandomAccessFile(leveldb_impl_, *r);
    return status;
  }

  virtual leveldb::Status NewWritableFile(const std::string& f,
                                          leveldb::WritableFile** r) OVERRIDE {
    leveldb::Status status = target()->NewWritableFile(f, r);
    if (*r != NULL) *r = new LevelDBWritableFile(leveldb_impl_, *r);
    return status;
  }

  virtual leveldb::Status DeleteFile(const std::string& f) OVERRIDE {
    struct stat st;
    memset(&st, 0, sizeof(st));
    // XXX: The file length *might* not be as large as its actual content
    // because the directory metadata can be updated later than the appends.
    int ret = stat(f.c_str(), &st);
    if (ret == 0) leveldb_impl_->Delete(static_cast<uint64_t>(st.st_size));

    return target()->DeleteFile(f);
  }

 private:
  class LevelDBImpl* leveldb_impl_;
};

LevelDBImpl::LevelDBImpl(const LevelDBParams& params, std::vector<Stat>& stats)
    : params_(params), stats_(stats) {
  stats_.push_back(Stat());

  pthread_mutex_init(&stats_mutex_, NULL);
  read_ = 0;
  appended_ = 0;

  // Clean up old files.
  leveldb::DestroyDB("leveldb_files", leveldb::Options());

  options_ = new leveldb::Options();

  options_->create_if_missing = true;

  // Turn off Snappy.
  options_->compression = leveldb::CompressionType::kNoCompression;

  // Use our Env to gather statistics.
  options_->env = new LevelDBEnv(this);

  // Limit the max open file count.
  options_->max_open_files = 900;

  // Configure the write buffer size.
  options_->write_buffer_size = params.log_size_threshold;

  // Do not overload insert.
  // These are hardcoded in leveldb/db/dbformat.h
  // options_->level0_file_num_compaction_trigger = 4;
  // options_->level0_slowdown_writes_trigger = 4;
  // options_->level0_stop_writes_trigger = 4;

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

  leveldb::Status status = leveldb::DB::Open(*options_, "leveldb_files", &db_);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }

  memset(value_buf_, 0, sizeof(value_buf_));
}

LevelDBImpl::~LevelDBImpl() {
  delete db_;

  delete options_->env;
  if (params_.use_custom_sizes) delete[] options_->custom_level_sizes;
  delete options_;

  pthread_mutex_destroy(&stats_mutex_);
}

void LevelDBImpl::print_status() const {
  // Force updating stats.
  const_cast<LevelDBImpl*>(this)->Delete(0);
}

void LevelDBImpl::dump_state(FILE* fp) const {
  // TODO: Implement.
  (void)fp;
}

void LevelDBImpl::put(LevelDBKey key, uint32_t item_size) {
  // LevelDB includes the full SSTable file size during calculating the level
  // size;
  // we consider the average space overhead per item in LevelDB so that the
  // average stored size becomes similar to item_size.
  const uint32_t overhead = 18;

  leveldb::Slice s_key(reinterpret_cast<const char*>(&key), sizeof(key));
  uint32_t value_size =
      static_cast<uint32_t>(static_cast<std::size_t>(item_size) - sizeof(key)) -
      overhead;
  assert(value_size < sizeof(value_buf_));
  leveldb::Slice s_value(value_buf_, value_size);

  leveldb::Status status = db_->Put(leveldb::WriteOptions(), s_key, s_value);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }
}

void LevelDBImpl::del(LevelDBKey key) {
  leveldb::Slice s_key(reinterpret_cast<const char*>(&key), sizeof(key));

  leveldb::Status status = db_->Delete(leveldb::WriteOptions(), s_key);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }
}

uint64_t LevelDBImpl::get(LevelDBKey key) {
  leveldb::Slice s_key(reinterpret_cast<const char*>(&key), sizeof(key));
  std::string s_value;
  uint64_t value;

  leveldb::Status status = db_->Get(leveldb::ReadOptions(), s_key, &s_value);
  if (!status.ok()) {
    printf("%s\n", status.ToString().c_str());
    assert(false);
  }
  assert(s_value.size() >= sizeof(uint64_t));
  value = *reinterpret_cast<const uint64_t*>(s_value.data());
  return value;
}

void LevelDBImpl::force_compact() {
  db_->CompactRange(NULL, NULL);

  // Force stat update.
  Delete(0);
}

void LevelDBImpl::Read(std::size_t len) { __sync_fetch_and_add(&read_, len); }

void LevelDBImpl::Append(std::size_t len) {
  __sync_fetch_and_add(&appended_, len);
}

void LevelDBImpl::Delete(std::size_t len) {
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
