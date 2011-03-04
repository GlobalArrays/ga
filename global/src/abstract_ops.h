#ifndef _ABSTRACT_OPS_H_
#define _ABSTRACT_OPS_H_

/* abstract operations, 'regular' (reg) and 'complex' (cpl) */
#define assign_reg(a,b) (a) = (b)
#define assign_cpl(a,b) (a).real = (b).real; \
                        (a).imag = (b).imag
#define assign_zero_reg(a) (a) = 0
#define assign_zero_cpl(a) (a).real = 0; \
                           (a).imag = 0
#define assign_add_reg(a,b,c) (a) = ((b) + (c))
#define assign_add_cpl(a,b,c) (a).real = ((b).real + (c).real); \
                              (a).imag = ((b).imag + (c).imag)
#define assign_mul_constant_reg(a,b,c) (a) = ((b) * (c))
#define assign_mul_constant_cpl(a,b,c) (a).real = ((b) * (c).real); \
                                       (a).imag = ((b) * (c).imag)
#define add_assign_reg(a,b) (a) += (b)
#define add_assign_cpl(a,b) (a).real += (b).real; \
                            (a).imag += (b).imag
#define neq_zero_reg(a) (0 != (a))
#define neq_zero_cpl(a) (0 != (a).real || 0 != (a).imag)
#define eq_zero_reg(a) (0 == (a))
#define eq_zero_cpl(a) (0 == (a).real && 0 == (a).imag)

#endif /* _ABSTRACT_OPS_H_ */
