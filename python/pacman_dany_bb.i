/* File : example.i */
%module pydany_bb
%{
#include "pacman_bb.hpp"
#include "pacman_bb_utils.hpp"

%}

// typemaps.i is a built-in swig interface that lets us map c++ types to other
// types in our language of choice. We'll use it to map Eigen matrices to
// Numpy arrays.
%include <typemaps.i>
%include <std_vector.i>
%include <std_list.i>

// eigen.i is found in ../swig/ and contains specific definitions to convert
// Eigen matrices into Numpy arrays.
%include <eigen.i>

%template(vectorMatrix4d) std::vector< Eigen::Matrix<double, 4, 4> >;
%template(vectorMatrixXd) std::vector< Eigen::MatrixXd >;
%template(vectorVectorXd) std::vector< Eigen::VectorXd >;
%template(listBoxes) std::list< pacman::Box >;

// Since Eigen uses templates, we have to declare exactly which types we'd
// like to generate mappings for.
%eigen_typemaps(Eigen::VectorXd)
%eigen_typemaps(Eigen::MatrixXd)
%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<double, 4, 4>)
%eigen_typemaps(Eigen::Matrix<double, 2, 3>)


/* Wrap a function taking a pointer to a function */
%include "pacman_bb.hpp"
%include "pacman_bb_utils.hpp"
