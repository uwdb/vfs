#include <string>
#include <iostream>
#include <glog/logging.h>
#include "mount.h"

#include <thread>

std::string usage() {
    return "vfs [path] [mount]";
}

void catf() {
    sleep(1);
    system("mkdir ../../mount/wolf");
    system("cat ../visualroad1.rgb > ../../mount/wolf/960x540.rgb");
    printf("Done\n");
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    system("fusermount -u ../../mount");
    std::thread cat(catf);

    if(argc == 2)
        std::cout << usage();
    else
        vfs::Mount(argv[1], argv[2]).run();

    cat.detach();
}