/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */



#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include "vptree.h"
#include "sptree.h"
#include "tsne.h"
#include "sptree.cpp"


using namespace std;

template<typename T, int OUTDIM>// Perform t-SNE
int TSNE<T, OUTDIM>::run(T* X, int N, int D, T* Y, T perplexity, T theta, int rand_seed,
               bool skip_random_init, bool verbose, int max_iter, int stop_lying_iter, int mom_switch_iter) {

    int no_dims = OUTDIM;
    // Set random seed
    if (skip_random_init != true) {
      if(rand_seed > 0) {
          if (verbose)
            printf("Using random seed: %d\n", rand_seed);
          srand((unsigned int) rand_seed);
      } else {
          if (verbose)
            printf("Using 0xDEADBEEF as random seed...\n");
          srand(0xDEADBEEF);
      }
    }

    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) {
        if (verbose) {
            printf("Perplexity too large for the number of data points!\n");
        }
        return 1;
    }
    if (verbose) {
        printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
    }
    bool exact = (theta == .0) ? true : false;

    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
	T momentum = .5, final_momentum = .8;
	T eta = 200.0;

    // Allocate some memory
    T* dY    = (T*) malloc(N * no_dims * sizeof(T));
    T* uY    = (T*) malloc(N * no_dims * sizeof(T));
    T* gains = (T*) malloc(N * no_dims * sizeof(T));
    if(dY == NULL || uY == NULL || gains == NULL) {
        if (verbose) {
            printf("Memory allocation failed!\n");
        }
        return 1;
    }

    for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    if (verbose) {
        printf("Computing input similarities...\n");
    }

    start = clock();
    zeroMean(X, N, D);
    T max_X = .0;
    for(int i = 0; i < N * D; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
    }
    for(int i = 0; i < N * D; i++) X[i] /= max_X;

    // Compute input similarities for exact t-SNE
    T* P; unsigned int* row_P; unsigned int* col_P; T* val_P;
    if(exact) {

        // Compute similarities
        if (verbose) {
            printf("Exact?");
        }
        P = (T*) malloc(N * N * sizeof(T));
        if(P == NULL) {
            if (verbose) {
                printf("Memory allocation failed!\n");
            }
            return 1;
        }
        computeGaussianPerplexity(X, N, D, P, perplexity);

        // Symmetrize input similarities
        if (verbose) {
            printf("Symmetrizing...\n");
        }

        int nN = 0;
        for(int n = 0; n < N; n++) {
            int mN = (n + 1) * N;
            for(int m = n + 1; m < N; m++) {
                P[nN + m] += P[mN + n];
                P[mN + n]  = P[nN + m];
                mN += N;
            }
            nN += N;
        }
        T sum_P = .0;
        for(int i = 0; i < N * N; i++) sum_P += P[i];
        for(int i = 0; i < N * N; i++) P[i] /= sum_P;
    }

    // Compute input similarities for approximate t-SNE
    else {

        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity), verbose);

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        T sum_P = .0;
        for(int i = 0; i < (int)row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < (int)row_P[N]; i++) val_P[i] /= sum_P;
    }
    end = clock();

    // Lie about the P-values
    if(exact) { for(int i = 0; i < N * N; i++)        P[i] *= 12.0; }
    else {      for(int i = 0; i < (int)row_P[N]; i++) val_P[i] *= 12.0; }

	// Initialize solution (randomly)
  if (skip_random_init != true) {
  	for(int i = 0; i < N * no_dims; i++) Y[i] = randn<T>() * .0001;
  }

	// Perform main training loop
    if (verbose) {
        if(exact) printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
        else printf("Input similarities computed in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (T) row_P[N] / ((T) N * (T) N));
    }
    start = clock();

	for(int iter = 0; iter < max_iter; iter++) {

        // Compute (approximate) gradient
        if(exact) computeExactGradient(P, Y, N, dY);
        else computeGradient(P, row_P, col_P, val_P, Y, N, dY, theta);

        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
		zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
            else      { for(int i = 0; i < (int)row_P[N]; i++) val_P[i] /= 12.0; }
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            end = clock();
            T C = .0;
            if(exact) C = evaluateError(P, Y, N);
            else      C = evaluateError(row_P, col_P, val_P, Y, N, theta);  // doing approximate computation here!
            if (verbose) {
                if(iter == 0)
                    printf("Iteration %d: error is %f\n", iter + 1, C);
                else {
                    total_time += (float) (end - start) / CLOCKS_PER_SEC;
                    printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC);
                }
            }
			start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    if(exact) free(P);
    else {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }

    if (verbose) {
        printf("Fitting performed in %4.2f seconds.\n", total_time);
    }

    return 0;
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::computeGradient(T* P, unsigned int* inp_row_P, unsigned int* inp_col_P, T* inp_val_P, T* Y, int N, T* dC, T theta)
{

    // Construct space-partitioning tree on current map
    SPTree<T, OUTDIM>* tree = new SPTree<T, OUTDIM>(Y, N);

    // Compute all terms required for t-SNE gradient
    T sum_Q = .0;
    T* pos_f = (T*) calloc(N * OUTDIM, sizeof(T));
    T* neg_f = (T*) calloc(N * OUTDIM, sizeof(T));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);

    #pragma omp parallel for schedule(guided) reduction(+:sum_Q)
    for(int n = 0; n < N; n++) {
        sum_Q += tree->computeNonEdgeForces(n, theta, neg_f + n * OUTDIM);
    }

    // Compute final t-SNE gradient
    for(int i = 0; i < N * OUTDIM; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::computeExactGradient(T* P, T* Y, int N, T* dC) {

	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * OUTDIM; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    T* DD = (T*) malloc(N * N * sizeof(T));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, OUTDIM, DD);

    // Compute Q-matrix and normalization sum
    T* Q    = (T*) malloc(N * N * sizeof(T));
    if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    T sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	// Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) {
        int mD = 0;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                T mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                for(int d = 0; d < OUTDIM; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD +=OUTDIM;
		}
        nN += N;
        nD +=OUTDIM;
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}


// Evaluate t-SNE cost function (exactly)
template<typename T, int OUTDIM>
T TSNE<T, OUTDIM>::evaluateError(T* P, T* Y, int N) {

    // Compute the squared Euclidean distance matrix
    T* DD = (T*) malloc(N * N * sizeof(T));
    T* Q = (T*) malloc(N * N * sizeof(T));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, OUTDIM, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    T sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
            else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    T C = .0;
	for(int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
template<typename T, int OUTDIM>
T TSNE<T, OUTDIM>::evaluateError(unsigned int* row_P, unsigned int* col_P, T* val_P, T* Y, int N, T theta)
{

    // Get estimate of normalization term
    SPTree<T, OUTDIM>* tree = new SPTree<T, OUTDIM>(Y, N);
    T buff[OUTDIM];
    T sum_Q = .0;
    for(int n = 0; n < N; n++)  {
        sum_Q += tree->computeNonEdgeForces(n, theta, buff);
    }

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    T C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * OUTDIM;
        for(int i = row_P[n]; i < (int)row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * OUTDIM;
            for(int d = 0; d < OUTDIM; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < OUTDIM; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < OUTDIM; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    // Clean up memory
    delete tree;
    return C;
}


// Compute input similarities with a fixed perplexity
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::computeGaussianPerplexity(T* X, int N, int D, T* P, T perplexity) {

	// Compute the squared Euclidean distance matrix
	T* DD = (T*) malloc(N * N * sizeof(T));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
    int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		T beta = 1.0;
		T min_beta = -DBL_MAX;
		T max_beta =  DBL_MAX;
		T tol = 1e-5;
        T sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[nN + m];
			T H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			T Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
        nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::computeGaussianPerplexity(T* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, T** _val_P, T perplexity, int K, bool verbose) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (T*) calloc(N * K, sizeof(T));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    T* val_P = *_val_P;
    T* cur_P = (T*) malloc((N - 1) * sizeof(T));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    VpTree<DataPoint<T>, T, euclidean_distance>* tree = new VpTree<DataPoint<T>, T, euclidean_distance>();
    vector<DataPoint<T> > obj_X(N, DataPoint<T>(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint<T>(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    if (verbose) {
        printf("Building tree...\n");
    }
    vector<DataPoint<T> > indices;
    vector<T> distances;
    for(int n = 0; n < N; n++) {

        if (verbose) {
            if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);
        }

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        T beta = 1.0;
        T min_beta = -DBL_MAX;
        T max_beta =  DBL_MAX;
        T tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; T sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            T H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            T Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}


// Symmetrizes a sparse matrix
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, T** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    T* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < (int)row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < (int)row_P[col_P[i] + 1]; m++) {
                if((int)col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    T* sym_val_P = (T*) malloc(no_elem * sizeof(T));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if((int)col_P[m] == n) {
                    present = true;
                    if(n <= (int)col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= (int)col_P[i])) {
                offset[n]++;
                if((int)col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::computeSquaredEuclideanDistance(T* X, int N, int D, T* DD) {
    const T* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const T* XmD = XnD + D;
        T* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        T* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Makes data zero-mean
template<typename T, int OUTDIM>
void TSNE<T, OUTDIM>::zeroMean(T* X, int N, int D) {

	// Compute data mean
	T* mean = (T*) calloc(D, sizeof(T));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (T) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
template<typename T>
T randn() {
	T x, y, radius;
	do {
		x = 2 * (rand() / ((T) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((T) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}




template<typename T>
int run_tSNE(T *inputData, T *outputData, int N, int in_dims, int out_dims, T theta, T perplexity, int rand_seed, bool verbose) {

  if (out_dims == 2) {
	  return TSNE<T, 2>::run(inputData, N, in_dims, outputData, perplexity, theta, rand_seed, false, verbose);
  } else if (out_dims == 3) {
    return TSNE<T, 3>::run(inputData, N, in_dims, outputData, perplexity, theta, rand_seed, false, verbose);
  } else {
    printf ("currently supports out_dims == 2 only");
    return 2;
  }
}
