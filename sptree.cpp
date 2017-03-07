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
#include <cmath>
#include "sptree.h"



// Constructs cell
template<typename T, int dimension>
Cell<T, dimension>::Cell() {}

template<typename T, int dimension>
Cell<T, dimension>::Cell(T* inp_corner, T* inp_width) {
    for(int d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
    for(int d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
}

// Destructs cell

template<typename T, int dimension>
Cell<T, dimension>::~Cell() {
}


template<typename T, int dimension>
T Cell<T, dimension>::getCorner(unsigned int d)  const{
    return corner[d];
}

template<typename T, int dimension>
T Cell<T, dimension>::getWidth(unsigned int d) const {
    return width[d];
}

template<typename T, int dimension>
void Cell<T, dimension>::setCorner(unsigned int d, T val) {
    corner[d] = val;
}

template<typename T, int dimension>
void Cell<T, dimension>::setWidth(unsigned int d, T val) {
    width[d] = val;
}

// Checks whether a point lies in a cell
template<typename T, int dimension>
bool Cell<T, dimension>::containsPoint(T point[]) const
{
    for(int d = 0; d < dimension; d++) {
        if(corner[d] - width[d] > point[d]) return false;
        if(corner[d] + width[d] < point[d]) return false;
    }
    return true;
}


// Default constructor for SPTree -- build tree, too!
template<typename T, int dimension>
SPTree<T, dimension>::SPTree(T* inp_data, unsigned int N)
{
    unsigned int D = dimension;
    // Compute mean, width, and height of current map (boundaries of SPTree)
    int nD = 0;
    T mean_Y[dimension], min_Y[dimension], max_Y[dimension];

    for(unsigned int d = 0; d < D; d++)  min_Y[d] =  DBL_MAX;
    for(unsigned int d = 0; d < D; d++)  max_Y[d] = -DBL_MAX;

    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int d = 0; d < D; d++) {
            mean_Y[d] += inp_data[n * D + d];
            if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
            if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < (int)D; d++) mean_Y[d] /= (T) N;

    // Construct SPTree
    T width[dimension];
    for(int d = 0; d < (int)D; d++) width[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    init(NULL, inp_data, mean_Y, width);
    fill(N);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
template<typename T, int dimension>
SPTree<T, dimension>::SPTree(T* inp_data, unsigned int N, T* inp_corner, T* inp_width)
{
    init(NULL, inp_data, inp_corner, inp_width);
    fill(N);
}


// Constructor for SPTree with particular size (do not fill the tree)
template<typename T, int dimension>
SPTree<T, dimension>::SPTree(T* inp_data, T* inp_corner, T* inp_width)
{
    init(NULL, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent (do not fill tree)
template<typename T, int dimension>
SPTree<T, dimension>::SPTree(SPTree<T, dimension>* inp_parent, T* inp_data, T* inp_corner, T* inp_width) {
    init(inp_parent, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
template<typename T, int dimension>
SPTree<T, dimension>::SPTree(SPTree<T, dimension>* inp_parent, T* inp_data, unsigned int N, T* inp_corner, T* inp_width)
{
    init(inp_parent, inp_data, inp_corner, inp_width);
    fill(N);
}


// Main initialization function
template<typename T, int dimension>
void SPTree<T, dimension>::init(SPTree<T, dimension>* inp_parent, T* inp_data, T* inp_corner, T* inp_width)
{
    parent = inp_parent;
    int D = dimension;
    no_children = 2;
    for(int d = 1; d < D; d++) no_children *= 2;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;

    for(int d = 0; d < D; d++) boundary.setCorner(d, inp_corner[d]);
    for(int d = 0; d < D; d++) boundary.setWidth( d, inp_width[d]);

    children = (SPTree<T, dimension>**) malloc(no_children * sizeof(SPTree<T, dimension>*));
    for(unsigned int i = 0; i < no_children; i++) children[i] = NULL;

    for(int d = 0; d < D; d++) center_of_mass[d] = .0;

}


// Destructor for SPTree
template<typename T, int dimension>
SPTree<T, dimension>::~SPTree()
{
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i] != NULL) delete children[i];
    }
    free(children);
}


// Update the data underlying this tree
template<typename T, int dimension>
void SPTree<T, dimension>::setData(T* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
template<typename T, int dimension>
SPTree<T, dimension>* SPTree<T, dimension>::getParent()
{
    return parent;
}


// Insert a point into the SPTree
template<typename T, int dimension>
bool SPTree<T, dimension>::insert(unsigned int new_index)
{
    // Ignore objects which do not belong in this quad tree
    T* point = data + new_index * dimension;
    if(!boundary.containsPoint(point))
        return false;

    // Online update of cumulative size and center-of-mass
    cum_size++;
    T mult1 = (T) (cum_size - 1) / (T) cum_size;
    T mult2 = 1.0 / (T) cum_size;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[d] *= mult1;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[d] += mult2 * point[d];

    // If there is space in this quad tree and it is a leaf, add the object here
    if(is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }

    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for(unsigned int n = 0; n < size; n++) {
        bool duplicate = true;
        for(unsigned int d = 0; d < dimension; d++) {
            if(point[d] != data[index[n] * dimension + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if(any_duplicate) return true;

    // Otherwise, we need to subdivide the current cell
    if(is_leaf) subdivide();

    // Find out where the point can be inserted
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i]->insert(new_index)) return true;
    }

    // Otherwise, the point cannot be inserted (this should never happen)
    return false;
}


// Create four children which fully divide this cell into four quads of equal area
template<typename T, int dimension>
void SPTree<T, dimension>::subdivide() {

    // Create new children
    T new_corner[dimension];
    T new_width[dimension];
    for(unsigned int i = 0; i < no_children; i++) {
        unsigned int div = 1;
        for(unsigned int d = 0; d < dimension; d++) {
            new_width[d] = .5 * boundary.getWidth(d);
            if((i / div) % 2 == 1) new_corner[d] = boundary.getCorner(d) - .5 * boundary.getWidth(d);
            else                   new_corner[d] = boundary.getCorner(d) + .5 * boundary.getWidth(d);
            div *= 2;
        }
        children[i] = new SPTree<T, dimension>(this, data, new_corner, new_width);
    }

    // Move existing points to correct children
    for(unsigned int i = 0; i < size; i++) {
        bool success = false;
        for(unsigned int j = 0; j < no_children; j++) {
            if(!success) success = children[j]->insert(index[i]);
        }
        index[i] = -1;
    }

    // Empty parent node
    size = 0;
    is_leaf = false;
}


// Build SPTree on dataset
template<typename T, int dimension>
void SPTree<T, dimension>::fill(unsigned int N)
{
    for(unsigned int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
template<typename T, int dimension>
bool SPTree<T, dimension>::isCorrect()
{
    for(unsigned int n = 0; n < size; n++) {
        T* point = data + index[n] * dimension;
        if(!boundary.containsPoint(point)) return false;
    }
    if(!is_leaf) {
        bool correct = true;
        for(int i = 0; i < no_children; i++) correct = correct && children[i]->isCorrect();
        return correct;
    }
    else return true;
}



// Build a list of all indices in SPTree
template<typename T, int dimension>
void SPTree<T, dimension>::getAllIndices(unsigned int* indices)
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in SPTree
template<typename T, int dimension>
unsigned int SPTree<T, dimension>::getAllIndices(unsigned int* indices, unsigned int loc)
{

    // Gather indices in current quadrant
    for(unsigned int i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;

    // Gather indices in children
    if(!is_leaf) {
        for(int i = 0; i < no_children; i++) loc = children[i]->getAllIndices(indices, loc);
    }
    return loc;
}


template<typename T, int dimension>
unsigned int SPTree<T, dimension>::getDepth() {
    if(is_leaf) return 1;
    int depth = 0;
    for(unsigned int i = 0; i < no_children; i++) depth = fmax(depth, children[i]->getDepth());
    return 1 + depth;
}


// Compute non-edge forces using Barnes-Hut algorithm
template<typename T, int dimension>
T SPTree<T, dimension>::computeNonEdgeForces(unsigned int point_index, T theta, T neg_f[]) const
{
    T resultSum = 0;
    T localbuff[dimension];
    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return resultSum;

    // Compute distance between point and center-of-mass
    T D = .0;
    unsigned int ind = point_index * dimension;
    for(unsigned int d = 0; d < dimension; d++) localbuff[d] = data[ind + d] - center_of_mass[d];
    for(unsigned int d = 0; d < dimension; d++) D += localbuff[d] * localbuff[d];

    // Check whether we can use this node as a "summary"
    T max_width = 0.0;
    T cur_width;
    for(unsigned int d = 0; d < dimension; d++) {
        cur_width = boundary.getWidth(d);
        max_width = (max_width > cur_width) ? max_width : cur_width;
    }

    if(is_leaf || max_width / sqrt(D) < theta) {
        // Compute and add t-SNE force between point and current node
        D = 1.0 / (1.0 + D);
        T mult = cum_size * D;
        resultSum += mult;
        mult *= D;
        for(unsigned int d = 0; d < dimension; d++) neg_f[d] += mult * localbuff[d];
    }
    else {
        // Recursively apply Barnes-Hut to children
        for(unsigned int i = 0; i < no_children; i++) {
            resultSum += children[i]->computeNonEdgeForces(point_index, theta, neg_f);
        }
    }

    return resultSum;
}


// Computes edge forces
template<typename T, int dimension>
void SPTree<T, dimension>::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, T* val_P, int N, T pos_f[]) const
{
    #pragma omp parallel for schedule(static)
    for(int n = 0; n < N; n++) {
        unsigned int ind1 = n*dimension;

        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {

            T localbuff[dimension];

            // Compute pairwise distance and Q-value
            T D = 1.0;
            unsigned int ind2 = col_P[i] * dimension;
            for(unsigned int d = 0; d < dimension; d++) localbuff[d] = data[ind1 + d] - data[ind2 + d];
            for(unsigned int d = 0; d < dimension; d++) D += localbuff[d] * localbuff[d];
            D = val_P[i] / D;

            // Sum positive force
            for(unsigned int d = 0; d < dimension; d++) pos_f[ind1 + d] += D * localbuff[d];
        }

    }
}


// Print out tree
template<typename T, int dimension>
void SPTree<T, dimension>::print()
{
    if(cum_size == 0) {
        printf("Empty node\n");
        return;
    }

    if(is_leaf) {
        printf("Leaf node; data = [");
        for(int i = 0; i < size; i++) {
            T* point = data + index[i] * dimension;
            for(int d = 0; d < dimension; d++) printf("%f, ", point[d]);
            printf(" (index = %d)", index[i]);
            if(i < size - 1) printf("\n");
            else printf("]\n");
        }
    }
    else {
        printf("Intersection node with center-of-mass = [");
        for(int d = 0; d < dimension; d++) printf("%f, ", center_of_mass[d]);
        printf("]; children are:\n");
        for(int i = 0; i < no_children; i++) children[i]->print();
    }
}
