#include "acado_common.h"
#include "acado_auxiliary_functions.h"
#include "solver_holder.h"

#include <stdio.h>
#include <math.h>
#include "ros/ros.h"

/* Some convenient definitions. */
#define NX          ACADO_NX  /* Number of differential state variables.  = 2 */
#define NXA         ACADO_NXA /* Number of algebraic variables. */
#define NU          ACADO_NU  /* Number of control inputs. = 2 */
#define NOD         ACADO_NOD  /* Number of online data values. */

#define NY          ACADO_NY  /* Number of measurements/references on nodes 0..N - 1. = 4 */ 
#define NYN         ACADO_NYN /* Number of measurements/references on node N. */

#define N           ACADO_N   /* Number of intervals in the horizon. = 10*/ 

// #define NUM_STEPS   10        /* Number of real-time iterations. */
//#define VERBOSE     0         /* Show iterations: 1, silent: 0.  */

/* Global variables used by the solver. */
ACADOvariables acadoVariables;
ACADOworkspace acadoWorkspace;

//MPC_solver::MPC_solver() : num_steps(0) { }
MPC_solver::MPC_solver(int n): num_steps(n) {
	int i;
	/* Initialize the solver. */
	acado_initializeSolver();

	/* Initialize the states and controls. */
	for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[i] = 0.0;
	for (i = 0; i < NU * N; ++i)  acadoVariables.u[i] = 0.0;

	/* Initialize the measurements/reference. */
	for (i = 0; i < NY * N; ++i)  acadoVariables.y[i] = 0.0;
};

int MPC_solver::reinitialize(){
	int i;
	/* Initialize the solver. */
	acado_initializeSolver();

	/* Initialize the states and controls. */
	for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[ i ] = 0.0;
	for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.0;

	/* Initialize the measurements/reference. */
	for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
	return 0;
};

double * MPC_solver::solve_mpc(double input_arr[4]) {
	// input_arr parsing: 0-2: current theta, 2-4: goal theta

	double x0[] = {input_arr[0], input_arr[1]};
	double yN[] = {input_arr[2], input_arr[3]};
	
	printf("current: %f, %f\n", x0[0], x0[1]);
	printf("goal: %f, %f\n", yN[0], yN[1]);

	/* Some temporary variables. */
	int    i, j, iter;
	acado_timer t;
	
	for (i = 0; i < N; i++) 
	{
		acadoVariables.y[i*NY] = yN[0]-x0[0];
		acadoVariables.y[i*NY+1] = yN[1]-x0[0];
    }
    // terminal goal
	for (i = 0; i < NYN; ++i) acadoVariables.yN[i] = 0;
	for (i = 0; i < 2; ++i) acadoVariables.x0[i] = x0[i];

	acado_tic( &t );
	for(iter = 0; iter < num_steps; ++iter)
	{
		/* Prepare for the RTI step. */
		acado_preparationStep();
		/* Compute the feedback step. */
		acado_feedbackStep( );
	}
	real_t te = acado_toc( &t );
	real_t KKT_val = acado_getKKT();

	ROS_INFO("Time: %.3g ms; KKT = %.3e", 1e3 * te, KKT_val);

	static double joint_commands[4];
    for (i = 0; i < 2; ++i) joint_commands[i] = acadoVariables.u[i];
	joint_commands[2] = 1e3 * te;
	joint_commands[3] = KKT_val;

	// if (KKT_val > 0.01) {
	// 	acado_initializeSolver();
	// 	/* Initialize the states and controls. */
	// 	for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[i] = 0.0;
	// 	for (i = 0; i < NU * N; ++i)  acadoVariables.u[i] = 0.0;

	// 	/* Initialize the measurements/reference. */
	// 	for (i = 0; i < NY * N; ++i)  acadoVariables.y[i] = 0.0;
	// } else {
		// acado_shiftStates(2, 0, 0);
		// acado_shiftControls( 0 );
	// }
	return joint_commands;
}
