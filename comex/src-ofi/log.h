#ifndef LOG_H_
#define LOG_H_

#include <assert.h>
#include <ctype.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "env.h"

typedef enum log_level_t
{
    ERROR = 0,
    WARN,
    INFO,
    DEBUG,
    TRACE
} log_level_t;

#define GET_TID() syscall(SYS_gettid)

#define COMEX_OFI_LOG(log_lvl, fmt, ...)                                               \
  do {                                                                                 \
        if (log_lvl <= env_data.log_level)                                             \
        {                                                                              \
            char time_buf[20]; /*2016:07:21 14:47:39*/                                 \
            get_time(time_buf, 20);                                                    \
            switch (log_lvl)                                                           \
            {                                                                          \
                case ERROR:                                                            \
                {                                                                      \
                    printf("%s: ERROR: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),   \
                            __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
                    print_backtrace();                                                 \
                    break;                                                             \
                }                                                                      \
                case WARN:                                                             \
                {                                                                      \
                    printf("WARNING: (%ld): " fmt "\n", GET_TID(), ##__VA_ARGS__);     \
                    break;                                                             \
                }                                                                      \
                case INFO:                                                             \
                {                                                                      \
                    printf("(%ld):" fmt "\n", GET_TID(), ##__VA_ARGS__);               \
                    break;                                                             \
                }                                                                      \
                case DEBUG:                                                            \
                case TRACE:                                                            \
                {                                                                      \
                    printf("%s: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),          \
                            __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
                    break;                                                             \
                }                                                                      \
                default:                                                               \
                {                                                                      \
                    assert(0);                                                         \
                }                                                                      \
            }                                                                          \
            fflush(stdout);                                                            \
        }                                                                              \
  } while (0)

static void get_time(char* buf, size_t buf_size)
{
    time_t timer;
    struct tm* time_info = 0;
    time(&timer);
    time_info = localtime(&timer);
    assert(time_info);
    strftime(buf, buf_size, "%Y:%m:%d %H:%M:%S", time_info);
}

static void print_backtrace(void)
{
    int j, nptrs;
    void* buffer[100];
    char** strings;

    nptrs = backtrace(buffer, 100);
    printf("backtrace() returned %d addresses\n", nptrs);
    fflush(stdout);

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL)
    {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < nptrs; j++)
    {
        printf("%s\n", strings[j]);
        fflush(stdout);
    }
    free(strings);
}

#endif /* LOG_H_ */
