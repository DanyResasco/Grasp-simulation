/* File : example.i */
%module Exampletest_dany_bb
%{
#include "pacman_bb_test.h"

%}

/* Wrap a function taking a pointer to a function */
%include "pacman_bb_test.h"

