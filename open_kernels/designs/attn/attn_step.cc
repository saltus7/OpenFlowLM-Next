#include "attn.h"
extern "C" {
void attn_step(const kvout_t *__restrict Kt, const kvout_t *__restrict Vt, const ATTN_QT *__restrict qs,
               float *__restrict oacc, float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  attn_step_impl(Kt, Vt, qs, oacc, ml, pb ATTN_H0_ARG);
}
}
