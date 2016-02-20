#include "common.h"
#include "util.h"
#include "zipf.h"
#include "leveldb.h"
#include "leveldb_impl.h"
#include "rocksdb_impl.h"
#include "meshdb.h"
#include <sys/time.h>

enum class ActiveKeyMode {
  kEntire = 0,
  kClustered = 1,
  kScattered = 2,
};

enum class DependencyMode {
  kIndependent = 0,
  kClustered = 1,
  kScattered = 2,
  kSequential = 3,
};

/*
class ItemLifetimeInfo : public MeshDBItemLifetimeInfo {
public:
    ItemLifetimeInfo(const zipf_gen_state& zipf_state, std::size_t
num_unique_keys, const std::vector<uint32_t>& keys) {
        item_class_.resize(num_unique_keys);
        item_lifetime_.resize(num_unique_keys);
        class_lifetime_.resize(3);

        std::vector<double> item_prob;
        item_prob.resize(num_unique_keys);

        std::vector<double> class_prob_sum;
        std::vector<std::size_t> class_count;
        class_prob_sum.resize(3);
        class_count.resize(3);

        double prob_sum = 0;
        for (std::size_t i = 0; i < num_unique_keys; i++) {
            uint64_t key = static_cast<uint64_t>(keys[i]);
            double prob = zipf_prob(&zipf_state, i);
            item_prob[key] = prob;
            prob_sum += prob;
        }

        for (std::size_t key = 0; key < num_unique_keys; key++) {
            double prob = item_prob[key];
            uint64_t lifetime = static_cast<uint64_t>(prob_sum / prob);      //
inverse of the actual probability
            // if (key < 100)
            //     printf("prob=%lf prob_sum=%lf lifetime=%zu\n", prob,
prob_sum, lifetime);

            std::size_t item_class = 0;
            if (lifetime < 100000)
                item_class = 0;
            else if (lifetime < 500000)
                item_class = 1;
            else
                item_class = 2;

            item_class_[key] = item_class;
            item_lifetime_[key] = lifetime;

            class_prob_sum[item_class] += prob;
            class_count[item_class]++;
        }

        for (std::size_t i = 0; i < 3; i++) {
            class_lifetime_[i] = static_cast<uint64_t>(prob_sum /
(class_prob_sum[i] / static_cast<double>(class_count[i])));
            printf("class_count[%zu]=%zu\n", i, class_count[i]);
            printf("class_lifetime[%zu]=%zu\n", i, class_lifetime_[i]);
        }
    }

    virtual ~ItemLifetimeInfo() {}
    virtual std::size_t item_class(MeshDBKey key) { return item_class_[key]; }
    virtual uint64_t item_lifetime(MeshDBKey key) { return item_lifetime_[key];
}
    virtual uint64_t class_lifetime(std::size_t lifetime_class) { return
class_lifetime_[lifetime_class]; }

private:
    std::vector<std::size_t> item_class_;
    std::vector<uint64_t> item_lifetime_;
    std::vector<std::size_t> class_lifetime_;
};
*/

void print_stats(std::vector<Stat>& stats, uint64_t insert_bytes) {
  double wa_r_sum = 0.;
  double wa_w_sum = 0.;
  for (std::size_t i = 0; i < stats.size(); i++) {
    if (i == 0)
      printf("<log> stats\n");
    else
      printf("<level-%zu> stats\n", i - 1);
    stats[i].print_status();
    double wa_r = static_cast<double>(stats[i].read_bytes()) /
                  static_cast<double>(insert_bytes);
    double wa_w = static_cast<double>(stats[i].write_bytes()) /
                  static_cast<double>(insert_bytes);
    printf("WA_r: %5.2lf\n", wa_r);
    printf("WA_w: %5.2lf\n", wa_w);
    wa_r_sum += wa_r;
    wa_w_sum += wa_w;
  }
  printf("WA_r sum: %5.2lf\n", wa_r_sum);
  printf("WA_w sum: %5.2lf\n", wa_w_sum);
}

uint64_t get_usec() {
  struct timeval tv_now;
  gettimeofday(&tv_now, NULL);

  return (uint64_t)tv_now.tv_sec * 1000000UL + (uint64_t)tv_now.tv_usec;
}

template <class StoreType>
void test(const char* store_type_name, uint32_t num_unique_keys,
          ActiveKeyMode active_key_mode, DependencyMode dependency_mode,
          uint64_t num_requests, double theta,
          LevelDBCompactionMode compaction_mode, uint64_t wb_size,
          bool enable_fsync, bool use_custom_sizes,
          const std::vector<uint64_t>& dump_points) {
  // The number of unique keys.
  // uint32_t num_unique_keys = 2 * 1000 * 1000;
  // The item size.
  uint32_t item_size = 1000;
  // The number of requests.
  // uint64_t num_requests = 20 * 1000 * 1000;
  // The skew of key popularity.  -1. = uniform no ramdom; 0. = uniform; 0.99 =
  // skewed; 40. = one key
  // double theta = -1.;
  // double theta = 0.;
  // double theta = 0.99;

  printf("store_type=%s\n", store_type_name);
  printf("num_unique_keys=%u\n", num_unique_keys);
  printf("active_key_mode=%u\n", active_key_mode);
  printf("dependency_mode=%u\n", dependency_mode);
  printf("item_size=%u\n", item_size);
  printf("num_requests=%lu\n", num_requests);
  printf("theta=%lf\n", theta);
  printf("compaction_mode=%u\n", compaction_mode);
  printf("wb_size=%lu\n", wb_size);
  printf("enable_fsync=%s\n", enable_fsync ? "1" : "0");
  printf("use_custom_sizes=%s\n", use_custom_sizes ? "1" : "0");
  printf("\n");
  fflush(stdout);

  bool verbose = true;
  // bool verbose = false;

  // Generate keys.
  // Uses uint32_t instead of uint64_t to reduce cache pollution.
  // TODO: Use hashing instead of the shuffle key array.
  std::vector<uint32_t> keys;
  // assert(num_unique_keys < (1UL << 32));
  sequence(num_unique_keys, keys);
  // Comment this out to disable hashing.
  shuffle(keys);

  // Initialize request generation.
  zipf_gen_state zipf_state;
  zipf_init(&zipf_state, static_cast<uint64_t>(num_unique_keys), theta, 1);

  // ItemLifetimeInfo lifetime_info(zipf_state, num_unique_keys, keys);
  // for (std::size_t i = 0; i < 4; i++)
  //     printf("class_lifetime(%zu)=%lu\n", i,
  //     lifetime_info.class_lifetime(i));
  // printf("item_class(0)=%lu\n", lifetime_info.item_class(keys[0]));
  // printf("item_class(100)=%lu\n", lifetime_info.item_class(keys[100]));
  // printf("item_class(10000)=%lu\n", lifetime_info.item_class(keys[10000]));
  // printf("item_class(1000000)=%lu\n",
  // lifetime_info.item_class(keys[1000000]));

  // Main simulation.
  std::vector<Stat> stats;
  LevelDBParams params;
  params.compaction_mode = compaction_mode;
  params.log_size_threshold = wb_size;
  params.enable_fsync = enable_fsync;
  params.use_custom_sizes = use_custom_sizes;
  params.hint_num_unique_keys = num_unique_keys;
  params.hint_theta = theta;

  StoreType store(params, stats);

  // MeshDBParams params;
  // MeshDB store(params, stat, &lifetime_info);

  // std::size_t next_dump = 0;
  (void)dump_points;

  // const uint64_t request_batch_size = 1000000;  // for debugging
  const uint64_t request_batch_size = 10000000;
  uint64_t num_processed_requests;
  uint64_t start_t;

  start_t = get_usec();

  {
    printf("initial insertion of %u items\n\n", num_unique_keys);
    fflush(stdout);

    num_processed_requests = 0;
    uint32_t key = 0;
    while (num_processed_requests < static_cast<uint64_t>(num_unique_keys)) {
      uint64_t this_request_batch_size = request_batch_size;
      if (num_processed_requests + this_request_batch_size > num_unique_keys)
        this_request_batch_size = num_unique_keys - num_processed_requests;

      for (uint64_t i = 0; i < this_request_batch_size; i++) {
        // for sequential insert
        store.put(key, item_size);
        // for random insert
        // store.put(keys[key], item_size);
        key++;
      }
      num_processed_requests += this_request_batch_size;

      if (verbose) {
        printf("key %lu/%u inserted\n", num_processed_requests,
               num_unique_keys);
        store.print_status();
        print_stats(stats,
                    num_processed_requests * static_cast<uint64_t>(item_size));
        printf("\n");
        fflush(stdout);
      }
    }

    printf("key %lu/%u inserted\n", num_processed_requests, num_unique_keys);
    store.print_status();
    print_stats(stats,
                num_processed_requests * static_cast<uint64_t>(item_size));
    printf("\n");
    fflush(stdout);
  }

  printf("elapsed time: %.3lf seconds\n\n",
         (double)(get_usec() - start_t) / 1000000.);

  for (auto& stat : stats) stat.reset();

  // How small fraction of keys are being used in the main transaction?
  const uint32_t active_key_factor = 10;

  // How many keys are dependent to each other?
  const int dependency_factor = 10;

  // Reinitialize request generation.
  uint32_t num_active_keys;
  switch (active_key_mode) {
    case ActiveKeyMode::kEntire:
      num_active_keys = num_unique_keys;
      break;
    case ActiveKeyMode::kClustered:
      num_active_keys = num_unique_keys / active_key_factor;
      sequence(num_active_keys, keys);
      shuffle(keys);
      break;
    case ActiveKeyMode::kScattered:
      num_active_keys = num_unique_keys / active_key_factor;
      break;
    default:
      assert(false);
      return;
  }
  zipf_init(&zipf_state, static_cast<uint64_t>(num_active_keys), theta, 2);

  start_t = get_usec();

  {
    printf("main transaction of %lu requests\n\n", num_requests);
    fflush(stdout);

    num_processed_requests = 0;
    while (num_processed_requests < num_requests) {
      uint64_t this_request_batch_size = request_batch_size;
      if (num_processed_requests + this_request_batch_size > num_requests)
        this_request_batch_size = num_requests - num_processed_requests;

      // Process a batch of requests.
      switch (dependency_mode) {
        case DependencyMode::kIndependent: {
          for (uint64_t i = 0; i < this_request_batch_size; i++) {
            uint32_t key = keys[zipf_next(&zipf_state)];
            // uint32_t key = keys[static_cast<uint64_t>(rand()) %
            // num_unique_keys];
            // uint32_t key = static_cast<uint32_t>(rand() % num_unique_keys);
            store.put(key, item_size);

            /*
            if (next_dump < dump_points.size() && dump_points[next_dump] ==
            num_processed_requests + i + 1) {
                char filename[1024];
                snprintf(filename, 1024, "output_state_%lu.txt",
            dump_points[next_dump]);
                FILE* fp_state = fopen(filename, "wt");
                store.dump_state(fp_state);
                fclose(fp_state);
                next_dump++;
            }
            */
          }
        } break;

        case DependencyMode::kClustered: {
          this_request_batch_size =
              (this_request_batch_size + dependency_factor - 1) /
              dependency_factor * dependency_factor;
          for (uint64_t i = 0; i < this_request_batch_size;
               i += dependency_factor) {
            uint32_t key = keys[zipf_next(&zipf_state)];
            key = key / dependency_factor * dependency_factor;
            store.put(key, item_size);

            for (int j = 1; j < dependency_factor; j++) {
              key++;
              if (key >= num_unique_keys) key -= num_unique_keys;
              store.put(key, item_size);
            }
          }
        } break;

        case DependencyMode::kScattered: {
          const uint32_t key_skip = num_unique_keys / dependency_factor;
          this_request_batch_size =
              (this_request_batch_size + dependency_factor - 1) /
              dependency_factor * dependency_factor;
          for (uint64_t i = 0; i < this_request_batch_size;
               i += dependency_factor) {
            uint32_t key = keys[zipf_next(&zipf_state)];
            key = key % key_skip;
            store.put(key, item_size);

            for (int j = 1; j < dependency_factor; j++) {
              key += key_skip;
              if (key >= num_unique_keys) key -= num_unique_keys;
              store.put(key, item_size);
            }
          }
        } break;

        case DependencyMode::kSequential: {
          for (uint64_t i = 0; i < this_request_batch_size; i++) {
            uint32_t key = static_cast<uint32_t>((num_processed_requests + i) %
                                                 num_active_keys);
            store.put(key, item_size);
          }
        } break;

        default:
          assert(false);
          return;
      }
      num_processed_requests += this_request_batch_size;

      if (verbose) {
        printf("request %lu/%lu processed\n", num_processed_requests,
               num_requests);
        store.print_status();
        print_stats(stats,
                    num_processed_requests * static_cast<uint64_t>(item_size));
        printf("\n");
        fflush(stdout);
      }
    }

    printf("request %lu/%lu processed\n", num_processed_requests, num_requests);
    store.print_status();
    print_stats(stats,
                num_processed_requests * static_cast<uint64_t>(item_size));
    printf("\n");
    fflush(stdout);
  }

  printf("elapsed time: %.3lf seconds\n\n",
         (double)(get_usec() - start_t) / 1000000.);

  if (false) {
    printf("forcing compaction\n");
    fflush(stdout);
    store.force_compact();

    store.print_status();
    print_stats(stats,
                num_processed_requests * static_cast<uint64_t>(item_size));
    printf("\n");
    fflush(stdout);
  }

  /*
  {
      FILE* fp_state = fopen("output_state_final.txt", "wt");
      store.dump_state(fp_state);
      fclose(fp_state);
  }

  // Write the key probability file.
  {
      std::vector<double> prob;
      prob.resize(num_unique_keys);
      for (std::size_t i = 0; i < num_unique_keys; i++) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          prob[key] = zipf_prob(&zipf_state, i);
      }

      FILE* fp_prob = fopen("output_prob.txt", "wt");
      for (std::size_t i = 0; i < num_unique_keys; i++)
          fprintf(fp_prob, "prob:%lf\n", prob[i]);
      fclose(fp_prob);
  }
  */
}

int main(int argc, const char* argv[]) {
  if (argc < 11) {
    printf(
        "%s STORE-TYPE NUM-UNIQUE-KEYS ACTIVE-KEY-MODE DEPENDENCY-MODE "
        "NUM-REQUESTS ZIPF-THETA COMPACTION-MODE WB-SIZE ENABLE-FSYNC "
        "USE-CUSTOM-SIZES [DUMP-POINTS]\n",
        argv[0]);
    printf("STORE-TYPE: leveldb-sim, leveldb-impl, rocksdb-impl\n");
    printf("NUM-UNIQUE-KEYS: 1000000, ...\n");
    printf("ACTIVE-KEY-MODE: 0, 1, 2\n");
    printf("DEPENDENCY-MODE: 0, 1, 2, 3\n");
    printf("NUM-REQUESTS: 10000000, ...\n");
    printf("ZIPF-THETA: 0.00, 0.99, ...\n");
    printf("COMPACTION-MODE: 0, 1, 2, ...\n");
    printf("WB-SIZE: 4194304, ...\n");
    printf("ENABLE-FSYNC: 0, 1\n");
    printf("USE-CUSTOM-SIZES: 0, 1\n");
    return 1;
  }
  int store_type;
  if (strcmp(argv[1], "leveldb-sim") == 0)
    store_type = 0;
  else if (strcmp(argv[1], "leveldb-impl") == 0)
    store_type = 1;
  else if (strcmp(argv[1], "rocksdb-impl") == 0)
    store_type = 2;
  else {
    printf("invalid STORE-TYPE\n");
    return 1;
  }

  uint32_t num_unique_keys = static_cast<uint32_t>(atoi(argv[2]));
  ActiveKeyMode active_key_mode = static_cast<ActiveKeyMode>(atoi(argv[3]));
  DependencyMode dependency_mode = static_cast<DependencyMode>(atoi(argv[4]));
  uint64_t num_requests = static_cast<uint64_t>(atol(argv[5]));
  double theta = atof(argv[6]);
  LevelDBCompactionMode compaction_mode =
      static_cast<LevelDBCompactionMode>(atoi(argv[7]));
  uint64_t wb_size = static_cast<uint64_t>(atol(argv[8]));
  bool enable_fsync = atoi(argv[9]) != 0;
  bool use_custom_sizes = atoi(argv[10]) != 0;
  std::vector<uint64_t> dump_points;
  for (int i = 11; i < argc; i++)
    dump_points.push_back(static_cast<uint64_t>(atol(argv[i])));

  if (store_type == 0)
    test<LevelDB>("leveldb-sim", num_unique_keys, active_key_mode,
                  dependency_mode, num_requests, theta, compaction_mode,
                  wb_size, enable_fsync, use_custom_sizes, dump_points);
  else if (store_type == 1)
    test<LevelDBImpl>("leveldb-impl", num_unique_keys, active_key_mode,
                      dependency_mode, num_requests, theta, compaction_mode,
                      wb_size, enable_fsync, use_custom_sizes, dump_points);
  else if (store_type == 2)
    test<RocksDBImpl>("rocksdb-impl", num_unique_keys, active_key_mode,
                      dependency_mode, num_requests, theta, compaction_mode,
                      wb_size, enable_fsync, use_custom_sizes, dump_points);
  else
    assert(false);

  return 0;
}
