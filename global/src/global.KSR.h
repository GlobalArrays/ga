#define LOCK(g_a, proc, x)      _gspwt(x)
#define UNLOCK(g_a, proc, x)       _rsp(x)
#define UNALIGNED(x)    (((unsigned long) (x)) % sizeof(long))
typedef __align128 unsigned char subpage[128];

int    KSRbarrier_mem_req();
void   KSRbarrier(), KSRbarrier_init();



/* this macro has been copied from publicly available TCGMSG port to KSR */
#define Copy(src, dest, n) \
{ \
    register void *_s_ = (src); \
    register void *_d_ = (dest); \
    register long _length_ = (long) (n); \
    register long _i_, _l_; \
    register long _r00_, _r01_, _r02_, _r03_; \
    register long _r04_, _r05_, _r06_, _r07_; \
    register long _r08_, _r09_, _r10_, _r11_; \
    register long _r12_, _r13_, _r14_, _r15_; \
 \
    if (UNALIGNED(_s_) || UNALIGNED(_d_)) \
    { \
        _l_ = _length_ / 16; \
        for (_i_ = 0; _i_ < _l_; _i_++, _length_ -= 16) \
        { \
            _r00_ = *((char *)_s_ +  0); \
            _r01_ = *((char *)_s_ +  1); \
            _r02_ = *((char *)_s_ +  2); \
            _r03_ = *((char *)_s_ +  3); \
            _r04_ = *((char *)_s_ +  4); \
            _r05_ = *((char *)_s_ +  5); \
            _r06_ = *((char *)_s_ +  6); \
            _r07_ = *((char *)_s_ +  7); \
            _r08_ = *((char *)_s_ +  8); \
            _r09_ = *((char *)_s_ +  9); \
            _r10_ = *((char *)_s_ + 10); \
            _r11_ = *((char *)_s_ + 11); \
            _r12_ = *((char *)_s_ + 12); \
            _r13_ = *((char *)_s_ + 13); \
            _r14_ = *((char *)_s_ + 14); \
            _r15_ = *((char *)_s_ + 15); \
            ((char *)_s_) += 16; \
            *((char *)_d_ +  0) = _r00_; \
            *((char *)_d_ +  1) = _r01_; \
            *((char *)_d_ +  2) = _r02_; \
            *((char *)_d_ +  3) = _r03_; \
            *((char *)_d_ +  4) = _r04_; \
            *((char *)_d_ +  5) = _r05_; \
            *((char *)_d_ +  6) = _r06_; \
            *((char *)_d_ +  7) = _r07_; \
            *((char *)_d_ +  8) = _r08_; \
            *((char *)_d_ +  9) = _r09_; \
            *((char *)_d_ + 10) = _r10_; \
            *((char *)_d_ + 11) = _r11_; \
            *((char *)_d_ + 12) = _r12_; \
            *((char *)_d_ + 13) = _r13_; \
            *((char *)_d_ + 14) = _r14_; \
            *((char *)_d_ + 15) = _r15_; \
            ((char *)_d_) += 16; \
        } \
        for (_i_ = 0; _i_ < _length_; _i_++) \
            *((char *)_d_)++ = *((char *)_s_)++; \
    } \
    else \
    { \
        _l_ = _length_ / sizeof(subpage); \
        for (_i_ = 0; _i_ < _l_; _i_++, _length_ -= sizeof(subpage)) \
        { \
            _r00_ = *((long *)_s_ +  0); \
            _r01_ = *((long *)_s_ +  1); \
            _r02_ = *((long *)_s_ +  2); \
            _r03_ = *((long *)_s_ +  3); \
            _r04_ = *((long *)_s_ +  4); \
            _r05_ = *((long *)_s_ +  5); \
            _r06_ = *((long *)_s_ +  6); \
            _r07_ = *((long *)_s_ +  7); \
            _r08_ = *((long *)_s_ +  8); \
            _r09_ = *((long *)_s_ +  9); \
            _r10_ = *((long *)_s_ + 10); \
            _r11_ = *((long *)_s_ + 11); \
            _r12_ = *((long *)_s_ + 12); \
            _r13_ = *((long *)_s_ + 13); \
            _r14_ = *((long *)_s_ + 14); \
            _r15_ = *((long *)_s_ + 15); \
            ((char *)_s_) += sizeof(subpage); \
            *((long *)_d_ +  0) = _r00_; \
            *((long *)_d_ +  1) = _r01_; \
            *((long *)_d_ +  2) = _r02_; \
            *((long *)_d_ +  3) = _r03_; \
            *((long *)_d_ +  4) = _r04_; \
            *((long *)_d_ +  5) = _r05_; \
            *((long *)_d_ +  6) = _r06_; \
            *((long *)_d_ +  7) = _r07_; \
            *((long *)_d_ +  8) = _r08_; \
            *((long *)_d_ +  9) = _r09_; \
            *((long *)_d_ + 10) = _r10_; \
            *((long *)_d_ + 11) = _r11_; \
            *((long *)_d_ + 12) = _r12_; \
            *((long *)_d_ + 13) = _r13_; \
            *((long *)_d_ + 14) = _r14_; \
            *((long *)_d_ + 15) = _r15_; \
            ((char *)_d_) += sizeof(subpage); \
        } \
        _l_ = _length_ / sizeof(long); \
        for (_i_ = 0; _i_ < _l_; _i_++, _length_ -= sizeof(long)) \
            *((long *)_d_)++ = *((long *)_s_)++; \
        for (_i_ = 0; _i_ < _length_; _i_++) \
            *((char *)_d_)++ = *((char *)_s_)++; \
    } \
}

