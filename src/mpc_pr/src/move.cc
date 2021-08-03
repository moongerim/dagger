
#include "mpc_pr/move.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "std_msgs/Int32.h"

using namespace std;

int rti_num = 5;

MPC_solver myMpcSolver(rti_num);

// Check feasibility
double config_space(double theta_1, double theta_2){
  double l1 = 0.5;
  double l2 = 0.4;
  double R_S = 0.2;
  double R_quad = (l2/8+R_S)*(l2/8+R_S)-0.001;
  double s1 = sin(theta_1);
  double c1 = cos(theta_1);
  double s12 = sin(theta_1+theta_2);
  double c12 = cos(theta_1+theta_2);
  double O11 = -0.6;
  double O12 = 0.7;
  double O21 = 0.6;
  double O22 = 0.7;
  double x1 = l1*s1+0.1;
  double x2 = l1*s1+l2*s12+0.1;
  double t1 = (l1*c1+1*l2*c12/8-O11)*(l1*c1+1*l2*c12/8-O11)+(l1*s1+1*l2*s12/8-O12)*(l1*s1+1*l2*s12/8-O12)-R_quad;
  double t2 = (l1*c1+3*l2*c12/8-O11)*(l1*c1+3*l2*c12/8-O11)+(l1*s1+3*l2*s12/8-O12)*(l1*s1+3*l2*s12/8-O12)-R_quad;
  double t3 = (l1*c1+5*l2*c12/8-O11)*(l1*c1+5*l2*c12/8-O11)+(l1*s1+5*l2*s12/8-O12)*(l1*s1+5*l2*s12/8-O12)-R_quad;
  double t4 = (l1*c1+7*l2*c12/8-O11)*(l1*c1+7*l2*c12/8-O11)+(l1*s1+7*l2*s12/8-O12)*(l1*s1+7*l2*s12/8-O12)-R_quad;
  double t5 = (l1*c1+1*l2*c12/8-O21)*(l1*c1+1*l2*c12/8-O21)+(l1*s1+1*l2*s12/8-O22)*(l1*s1+1*l2*s12/8-O22)-R_quad;
  double t6 = (l1*c1+3*l2*c12/8-O21)*(l1*c1+3*l2*c12/8-O21)+(l1*s1+3*l2*s12/8-O22)*(l1*s1+3*l2*s12/8-O22)-R_quad;
  double t7 = (l1*c1+5*l2*c12/8-O21)*(l1*c1+5*l2*c12/8-O21)+(l1*s1+5*l2*s12/8-O22)*(l1*s1+5*l2*s12/8-O22)-R_quad;
  double t8 = (l1*c1+7*l2*c12/8-O21)*(l1*c1+7*l2*c12/8-O21)+(l1*s1+7*l2*s12/8-O22)*(l1*s1+7*l2*s12/8-O22)-R_quad;
  
  double answer = 0.0;
  if (x1>0 and x2>0 and t1>0 and t2>0 and t3>0 and t4>0 and t5>0 and t6>0 and t7>0 and t8>0 and abs(theta_2)<1.57){
     answer = 1.0;
  }
  return answer;
}

// Introduce class to make safer goal change
class GoalFollower 
{ 
    // Access specifier 
    public: 
    // Data Members 
    ros::Publisher chatter_pub;
    ros::Publisher ee_pub;
    double goal[2] = {3.14, 0.0};
    double flag = 1.0;
    double comand_vel[2] = {0.0000, 0.0000};
    double dist[10] = {0.0000, 0.0000,0.0000, 0.0000,0.0000, 0.0000,0.0000, 0.0000,0.0000, 0.0000};
    double joint_position[2] = {0.0000, 0.0000};
    double joint_speed[2] = {0.0000, 0.0000};
    void change_dist_msg(const std_msgs::Float64MultiArray msg){ 
      for (int i=0; i<10; i++) dist[i] = msg.data[i];
    }
    
    void SendVelocity(const std_msgs::Float64MultiArray joint_vel_values){
    	chatter_pub.publish(joint_vel_values);
	    return;
    }

    void change_states_msg(const std_msgs::Float64MultiArray msg){ 
      for (int i=0; i<2; i++){
        joint_position[i] = msg.data[i];
        joint_speed[i] = msg.data[i+2];
      }
    }
    void change_goal_msg(const std_msgs::Float64MultiArray msg){ 
      flag = msg.data[2];
    }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "pr_controller");
  ros::NodeHandle n;
  ROS_INFO("Node Started");

  GoalFollower my_follower;
  my_follower.chatter_pub = n.advertise<std_msgs::Float64MultiArray>("/MPC_solutions", 1);
  ros::Subscriber joint_status = n.subscribe("/pr/joint_states", 1, &GoalFollower::change_states_msg, &my_follower);
  ros::Subscriber dist_status = n.subscribe("/CoppeliaSim/distances", 1, &GoalFollower::change_dist_msg, &my_follower);
  ros::Subscriber goal_status = n.subscribe("/pr/command", 1, &GoalFollower::change_goal_msg, &my_follower);

  std_msgs::Float64MultiArray joint_vel_values;
  double answer = 0.0;
  ros::Rate loop_rate(20);
  while (ros::ok()){
    double currentState_targetValue[4];
    double* solutions;
    answer = config_space(my_follower.joint_position[0],my_follower.joint_position[1]);
    if (answer<1.0){
      // If unfeasibility happens, we need to force to break the iteration
      joint_vel_values.data.clear();
      for (int i = 0; i < 4; i++) joint_vel_values.data.push_back(1000);
      my_follower.SendVelocity(joint_vel_values);
      printf("Unfeasible case\n");
      // Then reinitiatilize the solver:
      myMpcSolver.reinitialize();
      printf("-------NEW EPISODE----\n");
      sleep(1);
    }
    else{
      if(my_follower.flag==0){
        // If python iteration needs to reset the environment, MPC also needs to be reinitialized.
        myMpcSolver.reinitialize();
        printf("-------NEW EPISODE----\n");
        joint_vel_values.data.clear();
        for (int i = 0; i < 2; i++) joint_vel_values.data.push_back(0);
        for (int i = 0; i < 2; i++) joint_vel_values.data.push_back(my_follower.joint_position[i]);
        my_follower.SendVelocity(joint_vel_values);
      }else{
        for (int i = 0; i < 2; i++){
          currentState_targetValue[i] = my_follower.joint_position[i];
          currentState_targetValue[i+2] = my_follower.goal[i];
        }
        solutions = myMpcSolver.solve_mpc(currentState_targetValue);
        joint_vel_values.data.clear();
        for (int i = 0; i < 2; i++) joint_vel_values.data.push_back(solutions[i]);
        for (int i = 0; i < 2; i++) joint_vel_values.data.push_back(my_follower.joint_position[i]);
        printf("Solutions = %f, %f\n", solutions[0], solutions[1]);
        my_follower.SendVelocity(joint_vel_values);
      }
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}

