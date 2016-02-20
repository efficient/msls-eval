#include "common.h"
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>

uint64_t get_usec() {
  struct timeval tv_now;
  gettimeofday(&tv_now, NULL);

  return (uint64_t)tv_now.tv_sec * 1000000UL + (uint64_t)tv_now.tv_usec;
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    printf("%s PATH\n", argv[0]);
  }

  // write 1 GiB
  const size_t stride_len = 4 * 1048576;
  const size_t stride_count = 256;

  char* bytes = new char[stride_len];
  char* p;

  int fd = open(argv[1], O_CREAT | O_RDWR | O_TRUNC, 0644);
  double rw_cost_sum = 0.;

  for (int i = 0; i < 10; i++) {
    uint64_t start_t;
    double elapsed;
    size_t remaining_len;

    printf("seq %d\n", i + 1);
    fflush(stdout);

    memset(bytes, (i % 254) + 1, stride_len);

    // write data
    start_t = get_usec();
    lseek(fd, 0, SEEK_SET);
    for (size_t stride = 0; stride < stride_count; stride++) {
      p = bytes;
      remaining_len = stride_len;
      while (remaining_len > 0) {
        ssize_t wrote_len = write(fd, p, stride_len);
        if (wrote_len < 0) {
          perror("");
          close(fd);
          return 0;
        }
        p += wrote_len;
        remaining_len -= static_cast<size_t>(wrote_len);
      }
    }
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    elapsed = (double)(get_usec() - start_t) / 1000000.;
    double write_tput = (stride_len * stride_count) / elapsed;

    // read data
    start_t = get_usec();
    lseek(fd, 0, SEEK_SET);
    for (size_t stride = 0; stride < stride_count; stride++) {
      p = bytes;
      remaining_len = stride_len;
      while (remaining_len > 0) {
        ssize_t read_len = read(fd, p, remaining_len);
        if (read_len < 0) {
          perror("");
          close(fd);
          return 0;
        }
        p += read_len;
        remaining_len -= static_cast<size_t>(read_len);
      };
    }
    elapsed = (double)(get_usec() - start_t) / 1000000.;
    double read_tput = (stride_len * stride_count) / elapsed;

    double rw_cost = write_tput / read_tput;
    rw_cost_sum += rw_cost;

    printf("write tput = %7.2lf MB/s\n", write_tput / 1000000.);
    printf("read tput =  %7.2lf MB/s\n", read_tput / 1000000.);
    printf("r/w cost =   %7.3lf\n", rw_cost);
    printf("(avg) =      %7.3lf\n", rw_cost_sum / static_cast<double>(i + 1));
    printf("\n");
    fflush(stdout);
  }

  close(fd);
  return 0;
}
