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
 * Below is the statement from BanditTest.  This test shows that the bug is in the algo
 *
 * The algorithm implementation has problems when all context features are 0, as the expected reward will always become 0
 * as well. It is currently not apparent if this is a problem of the implementation of the algorithm itself.
 *
 * This test shows that the bug is in the algorithm itself, per the question in that statement
 *
 * When this test is run with context {0,1} & {1,0}, with the bug, the following are the results, showing the 0,1 context has all arms tried equally
 * Context 1 best arm should be 2
 * Context 2 best arm should be 3
 * Context ({0; 1}), Arm 1 mean payoff is: 0.11176763895119775 <-- Same
 * Context ({0; 1}), Arm 2 mean payoff is: 0.11176763895119775 <-- Same
 * Context ({0; 1}), Arm 3 mean payoff is: 0.11176763895119775 <-- Same
 * Context ({0; 1}), Arm 4 mean payoff is: 0.11176763895119775 <-- Same
 * Context ({1; 0}), Arm 1 mean payoff is: 0.8568426093164598
 * Context ({1; 0}), Arm 2 mean payoff is: 0.8557385633927821
 * Context ({1; 0}), Arm 3 mean payoff is: 0.8571325948469687
 * Context ({1; 0}), Arm 4 mean payoff is: 0.8554355147371019
 * Trials p/Arm (best arm should have highest trial count):
 * Context ({0; 1}), Arm 1 trial count is: 12507.0 <-- Same
 * Context ({0; 1}), Arm 2 trial count is: 12507.0 <-- Same
 * Context ({0; 1}), Arm 3 trial count is: 12507.0 <-- Same
 * Context ({0; 1}), Arm 4 trial count is: 12507.0 <-- Same
 * Context ({1; 0}), Arm 1 trial count is: 267.0
 * Context ({1; 0}), Arm 2 trial count is: 244.0
 * Context ({1; 0}), Arm 3 trial count is: 49189.0
 * Context ({1; 0}), Arm 4 trial count is: 272.0
 *
 *
 * When this test is run with context {0,1} & {1,0}, with out the bug, the following are the results, showing the 0,1 context is correct, arm 2 has highest trial count
 * Context 1 best arm should be 2
 * Context 2 best arm should be 3
 * Context ({0; 1}), Arm 1 mean payoff is: 0.8555965432625908
 * Context ({0; 1}), Arm 2 mean payoff is: 0.8558525865686305
 * Context ({0; 1}), Arm 3 mean payoff is: 0.8550557614645186
 * Context ({0; 1}), Arm 4 mean payoff is: 0.8546251192915426
 * Context ({1; 0}), Arm 1 mean payoff is: 0.8567006467150113
 * Context ({1; 0}), Arm 2 mean payoff is: 0.8564430307420418
 * Context ({1; 0}), Arm 3 mean payoff is: 0.8571095771029925
 * Context ({1; 0}), Arm 4 mean payoff is: 0.8567220875375925
 * Trials p/Arm (best arm should have highest trial count):
 * Context ({0; 1}), Arm 1 trial count is: 274.0
 * Context ({0; 1}), Arm 2 trial count is: 49166.0
 * Context ({0; 1}), Arm 3 trial count is: 301.0
 * Context ({0; 1}), Arm 4 trial count is: 287.0
 * Context ({1; 0}), Arm 1 trial count is: 300.0
 * Context ({1; 0}), Arm 2 trial count is: 263.0
 * Context ({1; 0}), Arm 3 trial count is: 49159.0
 * Context ({1; 0}), Arm 4 trial count is: 250.0
 *
 */
public class BanditTest_LinUCB_Context_Of_Zero {

  public static void main(String[] args) {
    int d = 2;
    int n = 4;
    double alpha = 12.5;

    LinUCB linUCB = new LinUCB(d, n, alpha);

    ArrayRealVector[] contexts = new ArrayRealVector[2];

    contexts[0] = new ArrayRealVector(new double[]{0,1});
    contexts[1] = new ArrayRealVector(new double[]{1,0});

    double[] context1ArmMeans = new double[]{0.1,0.8,0.1,0.1}; // Arm 2 has highest mean for context 1
    double[] context2ArmMeans = new double[]{0.1,0.1,0.8,0.1}; // Arm 3 has highest mean for context 1
    double[][] armMeans = new double[2][n];
    armMeans[0] = context1ArmMeans;
    armMeans[1] = context2ArmMeans;

    for (int j =0 ; j < d; j++) {
      for (int i = 0; i < n; i++) {
        System.out.println("Context (" + contexts[j] + "), Arm " + (i+1) + " mean is " + armMeans[j][i]);
      }
    }
    
    System.out.println("Context 1 best arm should be 2");
    System.out.println("Context 2 best arm should be 3");

    Random randomContext = new Random(7);
    Random randomReward  = new Random(8);

    int currentContext = 0;
    double reward = 0;
    double[][] armTrals = new double[2][n];

    for (int i = 0; i < 100000; i++) {
      // 50/50 on context
      currentContext = (randomContext.nextFloat() < .5) ? 0 : 1;
      int arm = linUCB.chooseArm(contexts[currentContext]);

      armTrals[currentContext][arm]++;

      // Reward based upon means
      reward = (randomReward.nextFloat() <= armMeans[currentContext][arm]) ? 1 : 0;
      linUCB.receiveReward(contexts[currentContext], arm, reward);
    }

    System.out.println("Results:");

    System.out.println("Means:");
    for (int j =0 ; j < d; j++) {
      double[] payoffs = linUCB.getPayoffs(contexts[j]);
      for (int i = 0; i < n; i++) {
        System.out.println("Context (" + contexts[j] + "), Arm " + (i+1) + " mean payoff is: " + payoffs[i]);
      }
    }

    System.out.println("Trials p/Arm (best arm should have highest trial count):");
    for (int j =0 ; j < d; j++) {
      for (int i = 0; i < n; i++) {
        System.out.println("Context (" + contexts[j] + "), Arm " + (i+1) + " trial count is: " + armTrals[j][i]);
      }
    }
  }
}
