/* VCS L-2016.06 + Ubuntu 18.04 (glibc 2.27) compatibility shim.
 *
 * Why: vcs1's `simp::GetVcsCmdOnCompile` -> `get_proc_stat` reads
 * /proc/<pid>/stat with a fixed-size buffer + fscanf. Modern Linux
 * /proc/<PID>/stat has 52 fields with values that overrun VCS's parser,
 * corrupting the FILE* and SIGSEGV-ing in fclose(). Symptom:
 *   "Got SIGSEGV ... Module SnpsSVA_classes / During post design resolution"
 *
 * Fix: intercept fopen("/proc/.../stat") via LD_PRELOAD and serve a
 * sanitized old-format string via fmemopen.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o vcs_shim.so vcs_shim.c -ldl
 *
 * Use:
 *   ulimit -s unlimited
 *   export LD_PRELOAD=/path/to/vcs_shim.so
 *   setarch x86_64 -R vcs -sverilog -assert svaext file.sv
 *
 * Container also needs --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
 * for setarch -R (ASLR disable) to work. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

static FILE* (*real_fopen)(const char*, const char*) = NULL;
static FILE* (*real_fopen64)(const char*, const char*) = NULL;

static const char *FAKE_STAT =
  "1 (vcs1) R 0 1 1 0 -1 4194560 1 0 0 0 0 0 0 0 20 0 1 0 1 4096 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0\n";

static int is_proc_stat(const char *p){
  if(!p) return 0;
  if(strncmp(p,"/proc/",6)!=0) return 0;
  size_t n=strlen(p);
  if(n>5 && strcmp(p+n-5,"/stat")==0) return 1;
  return 0;
}

FILE* fopen(const char* path, const char* mode){
  if(!real_fopen) real_fopen = dlsym(RTLD_NEXT, "fopen");
  if(is_proc_stat(path)){
    return fmemopen((void*)FAKE_STAT, strlen(FAKE_STAT), mode);
  }
  return real_fopen(path, mode);
}

FILE* fopen64(const char* path, const char* mode){
  if(!real_fopen64) real_fopen64 = dlsym(RTLD_NEXT, "fopen64");
  if(is_proc_stat(path)){
    return fmemopen((void*)FAKE_STAT, strlen(FAKE_STAT), mode);
  }
  if(real_fopen64) return real_fopen64(path, mode);
  return fopen(path, mode);
}
