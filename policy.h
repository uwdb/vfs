#ifndef VFS_POLICY_H
#define VFS_POLICY_H

#include <set>

class Policy {
public:
    Policy()
        : joint_{"/wolf"}
    {}

private:
    std::set<std::filesystem::path> joint_;
};

#endif //VFS_POLICY_H
