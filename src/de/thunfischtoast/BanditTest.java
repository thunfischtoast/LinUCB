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

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.Random;

/**
 * Small test class as inspired by John Maxwell (http://john-maxwell.com/post/2017-03-17/). We create a context of two
 * features that represent reading preferences of imaginary news site visitors. The visitors like or dislike sites with
 * sports content (context[0] = 1 or 0) and like or dislike sites with politics content (context[1] = 1 or 0). The
 * features are bound to integers for easier analysis. The bandit should offer one of three sites to the visitor, each
 * having different contents to offer for sports and politics.
 *
 * The algorithm implementation has problems when all context features are 0, as the expected reward will always become 0
 * as well. It is currently not apparent if this is a problem of the implementation of the algorithm itself.
 */
public class BanditTest {

    public static void main(String[] args) {
        HybridLinUCB linUCB = new HybridLinUCB(2, 2, 3, 18);
//        LinUCB linUCB = new LinUCB(2, 3, 5);

        Random random = new Random(7);

        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int i = 0; i < 3; i++) {
                    System.out.println("Arm " + i + ", Context (" + j + ", " + k + ") mean is " + getMean(i, new double[]{j, k}));
                }
            }
        }

        double maxReward = 0;
        double minReward = 1000;
        int[][][] counters = new int[3][2][2];

        for (int i = 0; i < 10000; i++) {
            double sports = random.nextInt(2);
            double politics = random.nextInt(2);

            ArrayRealVector context = new ArrayRealVector(new double[]{sports, politics});
            if(linUCB instanceof HybridLinUCB)
                context = context.append(context);

            int arm = linUCB.chooseArm(context);

            // make sure that rewards are between 0 and 1
            double reward = ((nextBoundedGaussian(random) + getMean(arm, context.toArray())) + 1 ) / 2.25;
            maxReward = Math.max(maxReward, reward);
            minReward = Math.min(minReward, reward);
            linUCB.receiveRewards(new RealVector[]{context}, new int[]{arm}, new double[]{reward});

            counters[arm][(int) sports][(int) politics]++;
        }

        System.out.println("Max reward is " + maxReward + " min is " + minReward);

        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                System.out.print("Chosen arm counts for context ("  + j + ", " + k + "): ");
                System.out.println(counters[0][j][k] + ", " + counters[1][j][k] + ", " + counters[2][j][k] + ", ");
            }
        }
    }

    private static double getMean(int arm, double[] context) {
        double sportsCoef;
        double politicsCoef;
        double armBaseline;

        if (arm == 0) {
            sportsCoef = 0.25;
            politicsCoef = 0.05;
            armBaseline = 0.025;
        } else if (arm == 1) {
            sportsCoef = 0.05;
            politicsCoef = 0.025;
            armBaseline = 0.05;
        } else {
            sportsCoef = 0.05;
            politicsCoef = 0.2;
            armBaseline = 0.075;
        }

        return armBaseline + context[0] * sportsCoef + context[1] * politicsCoef;
    }

    /**
     * Return a pseudorandom, Gaussian distributed double with mean 0 and standard deviation 1 bounded in [-1, 1]
     * @param random
     */
    private static double nextBoundedGaussian(Random random){
        double v = random.nextGaussian();
        v = Math.min(4, v);
        v = Math.max(-4, v);
        return v / 4;
    }

}
