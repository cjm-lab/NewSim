#include "new_connect_functions.hpp"

int* make_shuffled_arr(int max_val) {
    int* arr = new int[max_val];
    std::iota(arr, arr + max_val, 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(arr, arr+max_val, g);
    return arr;
}

float get_distance_wrapped(int Ax, int Ay, int Bx, int By, int Nx, int Ny) {
    // get absolute distance
    int dx = std::abs(Ax - Bx);
    int dy = std::abs(Ay - By);

    // determine if wrapping around the edge is shorter
    int shortest_dx = std::min(dx, Nx - dx);
    int shortest_dy = std::min(dy, Ny - dy);

    // calculate euclidean distance
    return std::sqrt((shortest_dx * shortest_dx) + (shortest_dy * shortest_dy));
}

void connectGOtoGO(int** GOtoGO_post, int GOConv, float recip,
                   int numGO, int GOx, int GOy, int span) {

    std::random_device rd;
    std::mt19937 gen(rd());

    int* num_pre_syn = new int[numGO];
    std::memset(num_pre_syn, 0, sizeof(int) * numGO);

    int num_recip_connects = round(GOConv * recip);

    // make all reciprocal connections first, each connection at a time
    std::cout << "Starting Recip..." << std::endl;
    int* GolRandom; 
    for (int cNum=0; cNum < num_recip_connects; cNum++) {
        GolRandom = make_shuffled_arr(numGO);
        for (int i=0; i<numGO; i++) {
            int GOpost = GolRandom[i];
            // make sure our cell hasnt already made a connection this round
            if (num_pre_syn[GOpost] <= cNum) {
                int posX = GOpost % GOx;
                int posY = GOpost / GOx;
                // get the cells whos distance relative to the pre is within span
                std::vector<int> pre_in_span;
                for (int c=0; c<numGO; c++) {
                    int preX = c % GOx;
                    int preY = c / GOx;
                    float dist = get_distance_wrapped(posX, posY, preX, preY, GOx, GOy);
                    if (dist <= span && dist != 0) 
                        pre_in_span.emplace_back(c);
                }
                int numCandidates = pre_in_span.size();
                std::shuffle(pre_in_span.begin(), pre_in_span.end(), gen); 
                // attempt to make connections by passing several checks
                for (int attempt=0; attempt<numCandidates; attempt++) {
                    int possiblePre = pre_in_span[attempt];
                    // check that its not already connected from the other cell
                    // and the pre cell hasnt exceeded number of recip connections
                    int* result = std::find(GOtoGO_post[GOpost], GOtoGO_post[GOpost] + (num_pre_syn[GOpost]), possiblePre);
                    if (result == (GOtoGO_post[GOpost] + (num_pre_syn[GOpost])) &&
                        num_pre_syn[possiblePre] < num_recip_connects) {
                        
                        // num_pre_syn always == num_post_syn during recip connections
                        // therefore can be treated as pre and post
                        GOtoGO_post[GOpost][num_pre_syn[GOpost]] = possiblePre;
                        GOtoGO_post[possiblePre][num_pre_syn[possiblePre]] = GOpost;
                        num_pre_syn[GOpost]++;
                        num_pre_syn[possiblePre]++;
                        break;
                    } 
                }
            }
        }
    }
    // check for cells that were unsuccessful in making enough recip connections
    std::vector<int> problems_idx, problems_num;
    int num_fixes_needed = 0;
    for (int i=0; i < numGO; i++) {
        if (num_pre_syn[i] < num_recip_connects) {
            problems_idx.emplace_back(i);
            problems_num.emplace_back(num_recip_connects - num_pre_syn[i]);
            num_fixes_needed += num_recip_connects - num_pre_syn[i];
        } else if (num_pre_syn[i] > num_recip_connects) {
            std::cout << "ERROR: Golgi cell " << i << " made " << num_pre_syn[i] << " connections." << std::endl;
        }
    }
    // if there are some that need to be fixed, iterate through them, get their distances to the others
    // and make that number of connections needed to their nearest neighbors
    if (problems_idx.size() > 0) {
        std::cout << "Fixing " << num_fixes_needed << " / " << problems_idx.size() << " (synapses/cells) to nearest neighbor..." << std::endl;
        for (int i=0; i<problems_idx.size(); i++) {
            int post = problems_idx[i];
            int postX = post % GOx;
            int postY = post / GOx;
            for (int j=0; j<problems_num[i]; j++) {
                // check that it hasnt been fulfilled passively 
                if (num_pre_syn[post] < num_recip_connects) {
                    int closest_cell;
                    float closest_dist = std::numeric_limits<float>::max(); 
                    for (int k=0; k<problems_idx.size(); k++) {
                        int preX = problems_idx[k] % GOx;
                        int preY = problems_idx[k] / GOx;
                        float dist = get_distance_wrapped(postX, postY, preX, preY, GOx, GOy);
                        if (dist < closest_dist && dist != 0 && num_pre_syn[problems_idx[k]] < num_recip_connects) {
                            // check that it hasnt already been connected to that cell
                            int* connected = std::find(GOtoGO_post[post], GOtoGO_post[post] + num_pre_syn[post], problems_idx[k]);
                            if (connected == GOtoGO_post[post] + num_pre_syn[post]) {
                                closest_cell = problems_idx[k];
                                closest_dist = dist;
                            }
                        }
                    }    
                    if (closest_dist == std::numeric_limits<float>::max()) {
                        std::cout << "Could not find connection for cell: " << post << std::endl;
                        assert(false);
                    }

                    // wire up based on that result
                    GOtoGO_post[post][num_pre_syn[post]] = closest_cell;
                    GOtoGO_post[closest_cell][num_pre_syn[closest_cell]] = post;
                    num_pre_syn[post]++;
                    num_pre_syn[closest_cell]++;
                } 
            }
        }
    }

    for (int i=0; i<numGO; i++) {
        if (num_pre_syn[i] > num_recip_connects) {
            std::cout << "Cell: " << i << " is over recip limit." << std::endl;
        } else if (num_pre_syn[i] < num_recip_connects) {
            std::cout << "Cell: " << i << " is under recip limit." << std::endl;
        }
    }
    // Now do standard connections
    // need to keep track of pre and post synaptic connections now
    int* num_post_syn = new int[numGO];
    for (int i=0; i<numGO; i++) 
        num_post_syn[i] = num_pre_syn[i];

    if (num_recip_connects < GOConv) {
        std::cout << "Starting std connections..." << std::endl;
        // make our inital attempts
        for (int cNum = num_recip_connects; cNum < GOConv; cNum++) {
            GolRandom = make_shuffled_arr(numGO);
            for (int i=0; i < numGO; i++) {
                int GOpost = GolRandom[i];
                int postX = GOpost % GOx;
                int postY = GOpost / GOx;
                // get our candidates in span
                std::vector<int> pre_in_span;
                for (int j=0; j<numGO; j++) {
                    int preX = j % GOx;
                    int preY = j / GOx;
                    float dist = get_distance_wrapped(postX, postY, preX, preY, GOx, GOy);
                    if (dist <= span && dist != 0) 
                        pre_in_span.emplace_back(j);
                }
                std::shuffle(pre_in_span.begin(), pre_in_span.end(), gen);
                // rank order pres in span by how many pre connections they've made
                // try to connect the lowest first
                int pre_syn_nums[pre_in_span.size()];
                for (int j=0; j < pre_in_span.size(); j++) {
                    int pre = pre_in_span[j];
                    pre_syn_nums[j] = num_pre_syn[pre];
                }
                std::vector<int> sorted_idx(pre_in_span.size());
                std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                std::stable_sort(sorted_idx.begin(), sorted_idx.end(), [&](int left, int right) {
                    return pre_syn_nums[left] < pre_syn_nums[right];
                });
                // go through the pres in span via the num_pre_syn ranked order and try to connect
                for (int j=0; j<pre_in_span.size(); j++) {
                    int pre = pre_in_span[sorted_idx[j]];
                    // check connection both ways so it doesn't accidentally become a recip
                    int* is_pre = std::find(GOtoGO_post[GOpost], GOtoGO_post[GOpost] + (num_post_syn[GOpost]), pre);
                    int* is_post = std::find(GOtoGO_post[pre], GOtoGO_post[pre] + (num_post_syn[pre]), GOpost);
                    // also check that it has pre synapses to make
                    if (is_pre == GOtoGO_post[GOpost] + (num_post_syn[GOpost]) &&
                        is_post == GOtoGO_post[pre] + (num_post_syn[pre]) &&
                        num_pre_syn[pre] < GOConv) {

                            GOtoGO_post[GOpost][num_post_syn[GOpost]] = pre;
                            num_post_syn[GOpost]++;
                            num_pre_syn[pre]++;
                            break;
                    }
                }
            }
        }

        // fix via checking pre and post arrays and wire up via nearest neighbors
        std::vector<int> pre_under, post_under;
        std::vector<int> pre_num, post_num;
        int pre_needed = 0;
        int post_needed = 0;
        for (int i=0; i<numGO; i++) {
            if (num_pre_syn[i] < GOConv) {
                pre_under.emplace_back(i);
                pre_num.emplace_back(GOConv - num_pre_syn[i]);
                pre_needed += GOConv - num_pre_syn[i];
            }
            if (num_post_syn[i] < GOConv) {
                post_under.emplace_back(i);
                post_num.emplace_back(GOConv - num_post_syn[i]);
                post_needed += GOConv - num_post_syn[i];
            }
        }
        if (pre_needed != post_needed) {
            std::cout << "Pre needed: " << pre_needed << " , Post needed: " << post_needed << std::endl;
        } else {
            std::cout << "Pre needed = Post needed" << std::endl;
        }
        // iterate through all the posts and find a pre based on closest distance
        // included ability for multiple needed connects if possible
        for (int i=0; i<post_under.size(); i++) {
            int post = post_under[i];
            int postX = post % GOx;
            int postY = post / GOx;
            for (int j=0; j<post_num[i]; j++) {
                int closest_cell;
                float closest_dist = std::numeric_limits<float>::max();
                for (int k=0; k<pre_under.size(); k++) {
                    int pre = pre_under[k];
                    int preX = pre % GOx;
                    int preY = pre / GOx;
                    float dist = get_distance_wrapped(postX, postY, preX, preY, GOx, GOy);

                    if (dist < closest_dist && dist != 0 
                        && num_pre_syn[pre] < GOConv) {
                        // check that it isnt connected the other way to avoid creating a recip
                        bool is_post = false;
                        for (int c=0; c<num_post_syn[pre]; c++) 
                            if (GOtoGO_post[pre][c] == post) is_post = true;
                        if (!is_post) {
                            closest_cell = pre;
                            closest_dist = dist;
                        }
                    }
                }
                if (closest_dist == std::numeric_limits<float>::max()) {
                    std::cout << "Could not find a presyn for cell: " << post << std::endl;
                    assert(false);
                }
                // wire up
                GOtoGO_post[post][num_post_syn[post]] = closest_cell;
                num_post_syn[post]++;
                num_pre_syn[closest_cell]++;
            }
        }
    }
    delete[] GolRandom;
}