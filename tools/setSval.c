/* -------------------------------------------------------------------------- */
/* SetSpv mexFunction */
/* -------------------------------------------------------------------------- */

#include "mex.h"
/* #include "blas.h" */
/*#include "matrix.h"*/

/* set values to sparse matrix S */
void mexFunction(int nargout, mxArray *pargout [], int nargin, const mxArray *pargin [])
{
    if (nargin != 3 || nargout != 1)
        mexErrMsgTxt ("Usage:  S = SetSpv ( S, v, L )") ;

    /* ---------------------------------------------------------------- */
    /* inputs */
    /* ---------------------------------------------------------------- */
    double *Sval = mxGetPr( pargin [0] );
    double *v    = mxGetPr( pargin [1] );
    double  LL   = mxGetScalar( pargin [2] );  
    unsigned long L = (unsigned long) LL;
    
    /* ---------------------------------------------------------------- */
    /* output */
    /* ---------------------------------------------------------------- */
    for (unsigned long k = 0; k < L; k++)
    {
        Sval[k] = v[k]; 
    }
    pargout[0] = pargin [0];
    return;
}
