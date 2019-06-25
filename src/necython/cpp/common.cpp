#include <random>
#include <thread>

namespace network_embedding {

using std::default_random_engine;
using std::thread;
using std::hash;

thread_local default_random_engine random_number_generator(clock() + hash<thread::id>()(std::this_thread::get_id()));

};