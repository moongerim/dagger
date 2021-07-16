#include "mpc_pr/move.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "std_msgs/Int32.h"

using namespace std;

ofstream myfile;
int rti_num = 10;

MPC_solver myMpcSolver(rti_num);

// Introduce class to make safer goal change
class GoalFollower 
{ 
    // Access specifier 
    public: 
    // Data Members 
    ros::Publisher chatter_pub;
    ros::Publisher ee_pub;
    double goal[2] = {3.14, 0.0};
    double comand_vel[2] = {0.0000, 0.0000};
    double joint_position[2] = {0.0000, 0.0000};
    double joint_speed[2] = {0.0000, 0.0000};
    int flag = 10;
    double dist[10] = {0.0000, 0.0000,0.0000, 0.0000,0.0000, 0.0000,0.0000, 0.0000,0.0000, 0.0000};
    // double flag = 1.0;
    void change_states_msg(const std_msgs::Float64MultiArray msg) 
    { 
       for (int i=0; i<2; i++) {
           joint_position[i] = msg.data[i];
           joint_speed[i] = msg.data[i+2];
       }
    }
    void change_dist_msg(const std_msgs::Float64MultiArray msg) 
    { 
       for (int i=0; i<10; i++) dist[i] = msg.data[i];
    }
    void change_flag_msg(const std_msgs::Float64MultiArray msg) 
    { 
       flag=msg.data[2];
    }
    void SendVelocity(const std_msgs::Float64MultiArray joint_vel_values){
    	chatter_pub.publish(joint_vel_values);
	    return;
    }
    void SendRef(const std_msgs::Float64MultiArray ee_values){
    	ee_pub.publish(ee_values);
	    return;
    }
}; 

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pr_controller");
  ros::NodeHandle n;
  ROS_INFO("Node Started");
  // string filename = "data"
  myfile.open("mpc_log/data_1.csv", ios::out); 
  GoalFollower my_follower;
  my_follower.chatter_pub = n.advertise<std_msgs::Float64MultiArray>("/MPC_solutions", 1);
  // my_follower.ee_pub = n.advertise<std_msgs::Float64MultiArray>("/reference", 1);
  ros::Subscriber joint_status = n.subscribe("/pr/joint_states", 1, &GoalFollower::change_states_msg, &my_follower);
  ros::Subscriber dist_status = n.subscribe("/CoppeliaSim/distances", 1, &GoalFollower::change_dist_msg, &my_follower);
  ros::Subscriber command = n.subscribe("/pr/command", 1, &GoalFollower::change_flag_msg, &my_follower);
  double init[2]={1.4,0.0000};
  int learning_iterator=0;
  int fileseq=2;
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    // MPC input:
    double currentState_targetValue[4];
    // answer=0;
    // Check if arrived
    float max_diff = 0;
    float temp = 0;
    for (int i = 0; i < 2; i++) {
        temp = abs(my_follower.goal[i] - my_follower.joint_position[i]);
        if (temp > max_diff) max_diff = temp; 
    }
    double* solutions;
    string filename;
    if (max_diff<0.1) {
        printf("Arrived\n");
        myfile.close();
        myMpcSolver.reinitialize();
        filename = "mpc_log/data_"+to_string(fileseq)+".csv";
        myfile.open(filename, ios::out); 
        fileseq++;
        sleep(5);
    }
    if (my_follower.flag==0){
        printf("-------------------NEW episode------------------\n");
        myfile.close();
        myMpcSolver.reinitialize();
        filename = "mpc_log/data_"+to_string(fileseq)+".csv";
        myfile.open(filename, ios::out); 
        fileseq++;
        sleep(5);
        learning_iterator = 0;
    }
    else{
      for (int i = 0; i < 2; i++){
        currentState_targetValue[i] = my_follower.joint_position[i];
        currentState_targetValue[i+2] = my_follower.goal[i];
      }
      solutions = myMpcSolver.solve_mpc(currentState_targetValue);
      std_msgs::Float64MultiArray joint_vel_values;
      joint_vel_values.data.clear();
      for (int i = 0; i < 2; i++) joint_vel_values.data.push_back(solutions[i]);
      for (int i = 0; i < 2; i++) joint_vel_values.data.push_back(my_follower.joint_position[i]);
      //joint_vel_values.data.push_back(1);
      printf("Solutions = %f, %f\n", solutions[0], solutions[1]);
      my_follower.SendVelocity(joint_vel_values);
    }
    if (myfile.is_open())
	  {
      myfile <<my_follower.joint_position[0]<<" "<<my_follower.joint_position[1]<<" ";
      myfile <<init[0]<<" "<<init[1]<<" "<<max_diff<<" ";
      myfile <<my_follower.joint_speed[0]<<" "<<my_follower.joint_speed[1]<<" ";
      myfile <<my_follower.dist[0]<<" "<<my_follower.dist[1]<<" "<<my_follower.dist[2]<<" "<<my_follower.dist[3]<<" ";
      myfile <<my_follower.dist[4]<<" "<<my_follower.dist[5]<<" "<<my_follower.dist[6]<<" "<<my_follower.dist[7]<<" ";
      myfile <<solutions[0]<<" "<<solutions[1]<<" "<<solutions[2]<<" "<<solutions[3]<<" "<<solutions[4]<< endl;
	  }
      else cout << "Unable to open file";
    
    if (my_follower.flag==1){
      learning_iterator++;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
  myfile.close();
  return 0;
}

