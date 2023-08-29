#pragma once
#include <vector>

bool Init_config();
bool dynamic_thread_loop();

struct None_Check {
    float checking_Version;
};

struct None_Move{
    int move_manner;
    std::string comx;
};

struct None_Fov{
    int fov_off;
    //int game_width;
    //int game_height;
    float game_HFOV;
    float game_VFOV;
    int game_x_pixel;
    int game_y_pixel;

};
struct None_Pid{
    int pid_off;
    float kp_x;
    float ki_x;
    float kd_x;
    float kp_y;
    float ki_y;
    float kd_y;
    int sample_num;
};

struct None_Aim{
    int range_top;
    int range_bottom;
    int range_left;
    int range_right;
    int cla_off;
    int label_chose;
    float move_offset;

};
struct None_Fire {
    int auto_off;
    int auto_method;
    int auto_sleep;
};

struct None_Key {
    int aim_off;
    int key_method;
    int button_key1;
    int button_key2;
    int end_key;
};

struct None_Pred {

    std::string engine_path;
    int frame;
    float conf;
    float iou;
    int sleep;
};
struct None_Windows {
    int capture_method;
    int win32_method;
    std::string win32_name;
    int show;
    float display_size;
};

struct None_Other {

    int console_refresh;
};

struct Config_Info {

    std::string Ini_Path;

    struct None_Check Check;
    struct None_Move Move;
    struct None_Fov Fov;
    struct None_Pid Pid;
    struct None_Aim Aim;
    struct None_Fire Fire;
    struct None_Key Key;
    struct None_Pred Pred;
    struct None_Windows Windows;
    struct None_Other Other;


    //节点
    std::vector<char*> Node = { "CHECK","MOVE","FOV","PID",
    "AIM","AUTO","KEY","PRED","WINDOW","OTHER" };

    //键值
    std::vector<char*> Check_val = { "checking_Version" };
    std::vector<char*> Move_val = { "move_manner","comx" };
    std::vector<char*> Fov_val = { "fov_off","game_HFOV","game_VFOV","game_x_pixel","game_y_pixel" };
    std::vector<char*> Pid_val = { "pid_off","kp_x","ki_x","kd_x","kp_y","ki_y","kd_y","sample_num"};
    std::vector<char*> Aim_val = { "range_top","range_bottom","range_left","range_right","cla_off",
    "label_chose","move_offset" };

    std::vector<char*> Fire_val = { "auto_off","auto_method","auto_sleep" };
    std::vector<char*> Key_val = { "aim_off","key_method","button_key1","button_key2","end_key" };
    std::vector<char*> Pred_val = { "engine_path ","frame","conf","iou","sleep" };
    std::vector<char*> Win_val = { "capture_method","win32_method","win32_name","show" };
    std::vector<char*> Other_val = { "console_refresh" };
};


