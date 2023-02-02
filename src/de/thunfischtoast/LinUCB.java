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

import java.util.Arrays;
import java.util.Random;

/**
 * This class implements a contextual bandit algorithm called LinUCB as proposed by Li, Langford and Schapire.
 * This is the version with disjoint linear models.
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
public class LinUCB {

    /** Context accumulators */
    protected RealMatrix[] A_a;
    protected RealMatrix[] A_a_inverse;

    /** Reward accumulators */
    protected RealMatrix[] b_a;

    /** Linear regression parameters */
    protected RealMatrix[] theta_hat_a;

    /** How many times the standard deviation of the expected payoff
     * is added to the predicted payoff in the ridge regression */
    protected double alpha;

    /** Number of features */
    protected int d;

    /** Number of arms */
    protected int n;

    /**
     * @param d     number of features
     * @param n     number of arms
     * @param alpha how many times the standard deviation of the expected payoff
     *              is added to the predicted payoff in the ridge regression
     */
    public LinUCB(int d, int n, double alpha) {
        if(d <= 0 || n <= 0)
            throw new IllegalArgumentException("Number of features and of arms must be > 0");

        A_a = new RealMatrix[n];
        A_a_inverse = new RealMatrix[n];
        b_a = new RealMatrix[n];
        theta_hat_a = new RealMatrix[n];
        this.alpha = alpha;
        this.d = d;
        this.n = n;

        RealMatrix i_d = MatrixUtils.createRealIdentityMatrix(d); // identity matrix

        for (int i = 0; i < n; i++) {
            A_a[i] = i_d.copy();

            // instantiate b_a as a 0 vector of length d
            b_a[i] = new Array2DRowRealMatrix(1, d);

            A_a_inverse[i] = MatrixUtils.inverse(A_a[i]);
            theta_hat_a[i] = A_a_inverse[i].multiply(b_a[i].transpose());
        }
    }

    /**
     * Receive a reward for the given context and arm. Update the regression parameters accordingly.
     */
    public void receiveReward(RealVector context, int arm, double reward) {
        receiveRewards(new RealVector[]{context}, new int[]{arm}, new double[]{reward});
    }

    /**
     * Receive multiple rewards for the given contexts and arms. Update the regression parameters accordingly.
     */
    public void receiveRewards(RealVector[] context, int[] arm, double[] reward) {
        if(context.length != arm.length || context.length != reward.length)
            throw new IllegalArgumentException("Must give the same number of contexts, arms and rewards");

        for (int i = 0; i < context.length; i++) {
            if(context[i].getDimension() != d)
                throw new IllegalArgumentException("Context does not have the same dimension as given in the constructor (" + d + ")");

            RealMatrix contextMatrix = new Array2DRowRealMatrix(context[i].toArray());
            RealMatrix contextMatrixTranspose = contextMatrix.transpose();

            RealMatrix xMultx_t = contextMatrix.multiply(contextMatrixTranspose);

            double[] contextMultipliedWithReward = context[i].mapMultiply(reward[i]).toArray();

            int curArm = arm[i];
            A_a[curArm] = A_a[curArm].add(xMultx_t); // update A[curArm] by adding x_t[curArm]*x_t[curArm]^transposed to it
            b_a[curArm] = b_a[curArm].add(new Array2DRowRealMatrix(contextMultipliedWithReward).transpose()); // update b[curArm] by adding r_t * x_t[curArm] to it

            A_a_inverse[curArm] = MatrixUtils.inverse(A_a[curArm]);
            theta_hat_a[curArm] = A_a_inverse[curArm].multiply(b_a[curArm].transpose());
        }
    }

    /**
     * @return the expected payoff for each arm for the given context
     */
    public double[] getPayoffs(RealVector context) {
        if(context.getDimension() != d)
            throw new IllegalArgumentException("Context does not have the same dimension as given in the constructor (" + d + ")");

        double[] payoffs = new double[n];

        RealMatrix x = new Array2DRowRealMatrix(context.toArray());
        RealMatrix x_t = x.transpose();

        for (int i = 0; i < n; i++) {
            RealMatrix secondProduct = x_t.multiply(A_a_inverse[i]).multiply(x);
            double secondElement = secondProduct.getEntry(0, 0);

            // RealMatrix firstProduct = theta_hat_a[i].multiply(x_t);   // Bug, all arms tried equally when context starts with a zero
            RealMatrix firstProduct = (theta_hat_a[i]).transpose().multiply(x); // Fixes bug in BanditTest_LinUCB_Context_Of_Zero test and matches the mathematical algo
            double firstElement = firstProduct.getEntry(0, 0);

            double secondElementSqTimesAlpha = alpha * Math.sqrt(Math.abs(secondElement));
            payoffs[i] = firstElement + secondElementSqTimesAlpha;
        }

        return payoffs;
    }

    /**
     * @return the arm which maximizes (expected payoff + (alpha * standard deviation)) for the given context.
     * If multiple arms do this choose randomly uniformly distributed among them.
     */
    public int chooseArm(RealVector context) {
        double[] payoffs = getPayoffs(context);

        double maxPayoff = Double.MAX_VALUE * -1;
        boolean[] viableArms = new boolean[n];
        Arrays.fill(viableArms, false);
        int numberOfViableArms = 0;
        int lastViableArm = -1;

        for (int i = 0; i < n; i++) {
            if (payoffs[i] > maxPayoff) {
                Arrays.fill(viableArms, false);
                viableArms[i] = true;
                numberOfViableArms = 1;
                maxPayoff = payoffs[i];
                lastViableArm = i;
            } else if (payoffs[i] == maxPayoff) {
                viableArms[i] = true;
                numberOfViableArms++;
                lastViableArm = i;
            }
        }

        if (numberOfViableArms == 0) {
            throw new RuntimeException("No viable arm!");
        } else if (numberOfViableArms == 1) {
            return lastViableArm;
        } else {
            // choose equally among viable arms
            int[] viableArmIndices = new int[numberOfViableArms];
            int counter = 0;
            for (int i = 0; i < viableArms.length; i++) {
                if (viableArms[i]) {
                    viableArmIndices[counter] = i;
                    counter++;
                }
            }

            Random random = new Random();

            return viableArmIndices[random.nextInt(viableArmIndices.length)];
        }
    }
}
