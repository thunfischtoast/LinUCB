/* Copyright (C) 2018 Christian Römer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact: https://github.com/thunfischtoast or christian.roemer[at]udo.edu
*/

package de.thunfischtoast;

import org.apache.commons.math3.linear.*;

/**
 * This class implements a contextual bandit algorithm called LinUCB as proposed by Li, Langford and Schapire.
 * This is the version with hybrid linear models.
 *
 * @inproceedings{li2010contextual,
 *   title={A contextual-bandit approach to personalized news article recommendation},
 *   author={Li, Lihong and Chu, Wei and Langford, John and Schapire, Robert E},
 *   booktitle={Proceedings of the 19th international conference on World wide web},
 *   pages={661--670},
 *   year={2010},
 *   organization={ACM}
 * }
 *
 * @author Christian Römer
 */
public class HybridLinUCB extends LinUCB {

    /** Linear regression parameters for shared model */
    private RealMatrix beta_hat;

    /** Context accumulators for shared model */
    private RealMatrix[] B_a;
    private RealMatrix A_0;

    /** Reward accumulator for shared model */
    private RealMatrix b_0;

    /** Number of shared features */
    private int k;

    /**
     * @param d     number of non-shared features
     * @param k     number of shared features
     * @param n     number of arms
     * @param alpha how many times the standard deviation of the expected payoff if added to the predicted payoff in the ridge regression
     */
    public HybridLinUCB(int d, int k, int n, double alpha) {
        super(d, n, alpha);

        if(k <= 0)
            throw new IllegalArgumentException("Number of features > 0. If there is are no shared features use @LinUCB");

        this.k = k;

        A_0 = MatrixUtils.createRealIdentityMatrix(k);
        b_0 = new Array2DRowRealMatrix(k, 1);

        beta_hat = MatrixUtils.inverse(A_0).multiply(b_0);

        B_a = new RealMatrix[n];
        for (int i = 0; i < n; i++) {
            B_a[i] = new Array2DRowRealMatrix(d, k);
        }
    }

    /**
     * Receive a reward for the given context and arm. Update the regression parameters accordingly.
     */
    public double[] getPayoffs(RealVector sharedContext, RealVector combinedContext){
        return getPayoffs(sharedContext.append(combinedContext));
    }

    /**
     * Receive a reward for the given context and arm. Update the regression parameters accordingly.
     * The given context must be of form [sharedContext,nonSharedContext].
     */
    @Override
    public double[] getPayoffs(RealVector combinedContext) {
        if(combinedContext.getDimension() != k + d)
            throw new IllegalArgumentException("The given context must be of form [sharedContext,nonSharedContext]!");

        RealVector sharedContext = combinedContext.getSubVector(0, k);
        RealVector nonSharedContext = combinedContext.getSubVector(k, d);

        double[] payoffs = new double[n];

        RealMatrix x = new Array2DRowRealMatrix(nonSharedContext.toArray());
        RealMatrix x_t = x.transpose();

        RealMatrix z = new Array2DRowRealMatrix(sharedContext.toArray());
        RealMatrix z_t = z.transpose();

        RealMatrix A_0_inv = MatrixUtils.inverse(A_0);

        for(int i = 0; i < n; i++){
            RealMatrix first = z_t.multiply(A_0_inv).multiply(z);
            RealMatrix second = z_t.multiply(A_a_inverse[i]).multiply(B_a[i].transpose()).multiply(A_a_inverse[i]).multiply(x).scalarMultiply(2);
            RealMatrix third = x_t.multiply(A_a_inverse[i]).multiply(x);
            RealMatrix fourth = x_t.multiply(A_a_inverse[i]).multiply(B_a[i]).multiply(A_0_inv).multiply(B_a[i].transpose()).multiply(A_a_inverse[i]).multiply(x);

            RealMatrix s = first.subtract(second).add(third).add(fourth);

            double firstElement = z_t.multiply(beta_hat).getEntry(0,0);
            double secondElement = x_t.multiply(theta_hat_a[i]).getEntry(0,0);

            if(firstElement != 0)
                payoffs[i] = firstElement + secondElement + (alpha * Math.sqrt(Math.abs(s.getEntry(0,0))));
            else
                payoffs[i] = firstElement + secondElement;
        }

        return payoffs;
    }

    /**
     * Receive multiple rewards for the given contexts and arms. Update the regression parameters accordingly.
     * The given contexts must be of form [sharedContext,nonSharedContext].
     */
    @Override
    public void receiveRewards(RealVector[] combinedContexts, int[] arm, double[] reward) {
        for (int i = 0; i < combinedContexts.length; i++) {
            RealVector combinedContext = combinedContexts[i];
            if(combinedContext.getDimension() != k + d)
                throw new IllegalArgumentException("The given context must be of form [sharedContext,nonSharedContext]!");

            RealVector sharedContext = combinedContext.getSubVector(0, k);
            RealVector nonSharedContext = combinedContext.getSubVector(k, d);

            RealMatrix sharedContextMatrix = new Array2DRowRealMatrix(sharedContext.toArray());
            RealMatrix sharedContextMatrixTranspose = sharedContextMatrix.transpose();
            RealMatrix nonSharedContextMatrix = new Array2DRowRealMatrix(nonSharedContext.toArray());
            RealMatrix nonSharedContextMatrixTranspose = nonSharedContextMatrix.transpose();

            RealMatrix zMultz_t = sharedContextMatrix.multiply(sharedContextMatrixTranspose);
            RealMatrix xMultx_t = nonSharedContextMatrix.multiply(nonSharedContextMatrixTranspose);
            RealMatrix xMultz_t = nonSharedContextMatrix.multiply(sharedContextMatrixTranspose);

            A_0 = A_0.add(B_a[arm[i]].transpose().multiply(A_a_inverse[arm[i]].transpose()).multiply(B_a[arm[i]]));

            b_0 = b_0.add(B_a[arm[i]].transpose().multiply(A_a_inverse[arm[i]].transpose()).multiply(b_a[arm[i]].transpose()));

            A_a[arm[i]] = A_a[arm[i]].add(xMultx_t); // update A[arm] by adding x_t[arm]*x_t[arm]^transposed to it
            B_a[arm[i]] = B_a[arm[i]].add(xMultz_t);

            double[] rMultx = nonSharedContext.mapMultiply(reward[i]).toArray();
            double[] rMultz = sharedContext.mapMultiply(reward[i]).toArray();
            b_a[arm[i]] = b_a[arm[i]].add(new Array2DRowRealMatrix(rMultx).transpose()); // update b[arm] by adding r_t * x_t[arm] to it

            A_0 = A_0.add(zMultz_t).subtract(B_a[arm[i]].transpose().multiply(A_a_inverse[arm[i]].multiply(B_a[arm[i]])));
            b_0 = b_0.add(new Array2DRowRealMatrix(rMultz)).subtract(B_a[arm[i]].transpose().multiply(A_a_inverse[arm[i]].multiply(b_a[arm[i]].transpose())));

            for (int j = 0; j < A_a.length; j++) {
                A_a_inverse[j] = MatrixUtils.inverse(A_a[j]);
                theta_hat_a[j] = A_a_inverse[j].multiply(b_a[j].transpose());
            }

            beta_hat = MatrixUtils.inverse(A_0).multiply(b_0);
        }
    }
}
