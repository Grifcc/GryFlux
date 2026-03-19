/*
 * Copyright (c) 2007 John Weaver
 * Copyright (c) 2015 Miroslav Krajicek
 * ... (版权和许可信息) ...
 */

#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include "matrix.h" // *** 确保这个文件存在并能被找到 ***

#include <list>
#include <utility>
#include <iostream>
#include <cmath>
#include <limits>


// *** 从你的 munkres.h 文件复制过来的 XYZMIN 和 XYZMAX 宏定义 ***
// (或者包含定义它们的头文件, 如果它们来自别处)
#ifndef XYZMIN
#define XYZMIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef XYZMAX
#define XYZMAX(a,b) (((a)>(b))?(a):(b))
#endif


template<typename Data> class Munkres
{
    static constexpr int NORMAL = 0;
    static constexpr int STAR   = 1;
    static constexpr int PRIME  = 2;
public:

    /**
     * @brief 求解线性分配问题 (匈牙利算法).
     * @param m 输入的代价矩阵 (会被修改!). 假设是行优先格式.
     * 求解后, 矩阵中值为 0 的位置表示最优分配, 其他位置为 -1.
     */
    void solve(Matrix<Data> &m) {
        // ... (你提供的 munkres.h 中的完整实现) ...
        const size_t rows = m.rows(),
                columns = m.columns(),
                size = XYZMAX(rows, columns);

        this->matrix = m;

        if ( rows != columns ) {
            matrix.resize(size, size, matrix.mmax());
        }

        mask_matrix.resize(size, size);

        row_mask = new bool[size];
        col_mask = new bool[size];
        for ( size_t i = 0 ; i < size ; i++ ) row_mask[i] = false;
        for ( size_t i = 0 ; i < size ; i++ ) col_mask[i] = false;

        replace_infinites(matrix);
        minimize_along_direction(matrix, rows >= columns);
        minimize_along_direction(matrix, rows <  columns);

        int step = 1;
        while ( step ) {
            switch ( step ) {
            case 1: step = step1(); break;
            case 2: step = step2(); break;
            case 3: step = step3(); break;
            case 4: step = step4(); break;
            case 5: step = step5(); break;
            }
        }

        for ( size_t row = 0 ; row < size ; row++ ) {
            for ( size_t col = 0 ; col < size ; col++ ) {
                if ( mask_matrix(row, col) == STAR ) {
                    matrix(row, col) = 0;
                } else {
                    matrix(row, col) = -1;
                }
            }
        }

        matrix.resize(rows, columns);
        m = matrix;

        delete [] row_mask;
        delete [] col_mask;
    }

    // ... (replace_infinites, minimize_along_direction 实现) ...
    static void replace_infinites(Matrix<Data> &matrix) {
      const size_t rows = matrix.rows(), columns = matrix.columns();
      double max_val = std::numeric_limits<Data>::lowest(); // Use lowest() for potentially negative numbers
      bool found_finite = false;
      constexpr auto infinity = std::numeric_limits<Data>::infinity();

      for ( size_t row = 0 ; row < rows ; row++ ) {
        for ( size_t col = 0 ; col < columns ; col++ ) {
          if ( matrix(row, col) != infinity && matrix(row, col) == matrix(row, col)) { // Check for NaN too
            if (!found_finite || matrix(row, col) > max_val) {
              max_val = matrix(row, col);
              found_finite = true;
            }
          }
        }
      }

      Data replacement_val;
      if (!found_finite) {
          replacement_val = 1; // Or some default if all are infinite/NaN
      } else {
          // Find a value larger than max_val that is representable
          if (max_val < std::numeric_limits<Data>::max()) {
              replacement_val = max_val + 1; // Simple increment if possible
          } else {
              replacement_val = std::numeric_limits<Data>::max(); // Use max possible value
          }
          // Avoid making it infinity again if max_val was already huge
          if (replacement_val == infinity) {
             replacement_val = max_val; // Fallback or handle differently
          }
      }


      for ( size_t row = 0 ; row < rows ; row++ ) {
        for ( size_t col = 0 ; col < columns ; col++ ) {
          if ( matrix(row, col) == infinity || matrix(row, col) != matrix(row, col)) { // Replace Inf and NaN
            matrix(row, col) = replacement_val;
          }
        }
      }
    }

    static void minimize_along_direction(Matrix<Data> &matrix, const bool over_columns) {
      const size_t outer_size = over_columns ? matrix.columns() : matrix.rows(),
                   inner_size = over_columns ? matrix.rows() : matrix.columns();

      for ( size_t i = 0 ; i < outer_size ; i++ ) {
        Data min_val = over_columns ? matrix(0, i) : matrix(i, 0);
        bool found_min = true; // Assume first element is initially the min

        for ( size_t j = 1 ; j < inner_size; j++ ) {
          Data current_val = over_columns ? matrix(j, i) : matrix(i, j);
          if (current_val < min_val) {
             min_val = current_val;
          }
        }

        if (found_min && min_val > 0) { // Subtract only if a valid minimum > 0 was found
          for ( size_t j = 0 ; j < inner_size ; j++ ) {
            if ( over_columns ) {
              matrix(j, i) -= min_val;
            } else {
              matrix(i, j) -= min_val;
            }
          }
        }
      }
    }


private:
    // ... (step1 to step5 和其他私有成员及方法的实现) ...
      inline bool find_uncovered_in_matrix(const Data item, size_t &row, size_t &col) const {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        for ( row = 0 ; row < rows ; row++ ) {
          if ( !row_mask[row] ) {
            for ( col = 0 ; col < columns ; col++ ) {
              if ( !col_mask[col] ) {
                // Use a tolerance comparison for floating point types
                 if constexpr (std::is_floating_point_v<Data>) {
                     if (std::fabs(matrix(row,col) - item) < std::numeric_limits<Data>::epsilon()) {
                       return true;
                     }
                 } else {
                     if ( matrix(row,col) == item ) {
                       return true;
                     }
                 }
              }
            }
          }
        }
        return false;
      }

      bool pair_in_list(const std::pair<size_t,size_t> &needle, const std::list<std::pair<size_t,size_t> > &haystack) {
        for ( auto const& item : haystack ) {
          if ( needle == item ) {
            return true;
          }
        }
        return false;
      }

      int step1() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        std::vector<bool> row_starred(rows, false);
        std::vector<bool> col_starred(columns, false);

        for ( size_t row = 0 ; row < rows ; row++ ) {
          for ( size_t col = 0 ; col < columns ; col++ ) {
             // Use tolerance for zero comparison with floats
             bool is_zero;
             if constexpr (std::is_floating_point_v<Data>) {
                 is_zero = (std::fabs(matrix(row, col)) < std::numeric_limits<Data>::epsilon());
             } else {
                 is_zero = (matrix(row, col) == 0);
             }

            if ( is_zero && !row_starred[row] && !col_starred[col] ) {
                  mask_matrix(row,col) = STAR;
                  row_starred[row] = true;
                  col_starred[col] = true;
             }
          }
        }
        return 2;
      }

      int step2() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        size_t covercount = 0;
        for ( size_t i = 0 ; i < columns ; i++ ) col_mask[i] = false; // Reset column masks

        for ( size_t row = 0 ; row < rows ; row++ ) {
          for ( size_t col = 0 ; col < columns ; col++ ) {
            if ( STAR == mask_matrix(row, col) ) {
              col_mask[col] = true; // Cover column if it has a starred zero
            }
          }
        }
         for ( size_t i = 0 ; i < columns ; i++ ) {
             if (col_mask[i]) covercount++;
         }


        if ( covercount >= matrix.minsize() ) { // Use minsize which handles non-square matrices
          return 0; // Done
        }
        return 3;
      }

      int step3() {
        while ( find_uncovered_in_matrix(0, saverow, savecol) ) { // Find an uncovered zero
          mask_matrix(saverow,savecol) = PRIME; // Prime it
          bool found_star_in_row = false;
          size_t star_col = 0;
          for ( size_t ncol = 0 ; ncol < matrix.columns() ; ncol++ ) {
            if ( mask_matrix(saverow,ncol) == STAR ) {
              found_star_in_row = true;
              star_col = ncol;
              break;
            }
          }

          if (!found_star_in_row) {
             // If no Z* in the row of Z', go to Step 4
             return 4;
          } else {
             // If Z* exists, cover this row and uncover column of Z*
             row_mask[saverow] = true;
             col_mask[star_col] = false;
             // Continue loop (Return to Step 3.1 in description)
          }
        }
        // If no uncovered zero exists, go to Step 5
        return 5;
      }

      int step4() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        std::list<std::pair<size_t,size_t> > seq;
        std::pair<size_t,size_t> z0(saverow, savecol); // Unpaired Z' from Step 3
        seq.push_back(z0);

        size_t current_row = saverow;
        size_t current_col = savecol;

        while (true) {
            // Find Z* in the column of Z[2N] (current Z')
            size_t star_row = -1;
            for (size_t r = 0; r < rows; ++r) {
                if (mask_matrix(r, current_col) == STAR) {
                    star_row = r;
                    break;
                }
            }

            if (star_row == static_cast<size_t>(-1)) {
                break; // Sequence terminates with an unpaired Z'
            }

            std::pair<size_t, size_t> z_star(star_row, current_col);
            seq.push_back(z_star);

            // Find Z' in the row of Z[2N+1] (current Z*)
            size_t prime_col = -1;
            for (size_t c = 0; c < columns; ++c) {
                if (mask_matrix(star_row, c) == PRIME) {
                    prime_col = c;
                    break;
                }
            }

            std::pair<size_t, size_t> z_prime(star_row, prime_col);
            seq.push_back(z_prime);

            current_row = star_row;
            current_col = prime_col;
         }

        // Augment the path
        for (const auto& pair : seq) {
          if (mask_matrix(pair.first, pair.second) == STAR)
            mask_matrix(pair.first, pair.second) = NORMAL;
          else // Must be PRIME
            mask_matrix(pair.first, pair.second) = STAR;
        }

        // Erase all primes
        for ( size_t r = 0 ; r < rows ; r++ ) {
          for ( size_t c = 0 ; c < columns ; c++ ) {
            if ( mask_matrix(r,c) == PRIME ) {
              mask_matrix(r,c) = NORMAL;
            }
          }
        }
        // Uncover all rows and columns
        for ( size_t i = 0 ; i < rows ; i++ ) row_mask[i] = false;
        for ( size_t i = 0 ; i < columns ; i++ ) col_mask[i] = false;

        return 2; // Return to Step 2
      }

      int step5() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        Data h = std::numeric_limits<Data>::max();
        bool found_uncovered = false;

        // Find the smallest uncovered value (h)
        for ( size_t row = 0 ; row < rows ; row++ ) {
          if ( !row_mask[row] ) {
            for ( size_t col = 0 ; col < columns ; col++ ) {
              if ( !col_mask[col] ) {
                 if (matrix(row, col) < h) {
                     h = matrix(row, col);
                     found_uncovered = true;
                 }
              }
            }
          }
        }

        if (!found_uncovered || h == std::numeric_limits<Data>::max()) {
           // Should not happen if matrix is finite, but handle defensively
           // Maybe indicate an error or break? For now, go back to step 3 hoping something changed.
           return 3;
        }

        // Add h to all covered rows
        for ( size_t row = 0 ; row < rows ; row++ ) {
          if ( row_mask[row] ) {
            for ( size_t col = 0 ; col < columns ; col++ ) {
              matrix(row, col) += h;
            }
          }
        }

        // Subtract h from all uncovered columns
        for ( size_t col = 0 ; col < columns ; col++ ) {
          if ( !col_mask[col] ) {
            for ( size_t row = 0 ; row < rows ; row++ ) {
              matrix(row, col) -= h;
            }
          }
        }

        return 3; // Return to Step 3
      }


      Matrix<int> mask_matrix;
      Matrix<Data> matrix;
      bool *row_mask = nullptr; // Initialize to nullptr
      bool *col_mask = nullptr; // Initialize to nullptr
      size_t saverow = 0, savecol = 0;
};


#endif /* !defined(_MUNKRES_H_) */