#include "new_connect_functions.hpp"

// Utility Functions

std::function<int()> make_generator(int min_val, int max_val) {
    // seed each engine differently with rd
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val-1);

    // return the engine as a lambda,
    // so it becomes unique for each function all
    return [engine, dist]() mutable {
        return dist(engine);
    };
}
 
int* make_shuffled_arr(int max_val) {
    int* arr = new int[max_val];
    std::iota(arr, arr + max_val, 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(arr, arr+max_val, g);
    return arr;
}

std::vector<int> make_candidate_vector_from_2D_range(int posX, int posY, int spanX, int spanY, int numX, int numY) {
    std::vector<int> vector;
    vector.reserve(spanX*spanY);

    int startX = (posX - spanX/2);
    int startY = (posY - spanY/2);

    // itterate over span
    for (int i = 0; i < spanY; i++) {
        int y_raw = i + startY;
        // wrap position over edges of grid, this funky code wraps over negatives too
        int y = ((y_raw % numY) + numY) % numY;

        for (int j = 0; j < spanX; j++) {
            int x_raw = j + startX;
            // both ways wrapping
            int x = ((x_raw % numX) + numX) % numX;
            // flatten and add
            int num = (y * numX) + x;
            vector.push_back(num);
        }
    }

    return vector;
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

// Timer helper struct
void Timer::start() {
    startTime = std::chrono::high_resolution_clock::now();
}
void Timer::stop() {
    endTime = std::chrono::high_resolution_clock::now();
}
void Timer::print() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    if (timeFrame == "ms") {
        std::cout << message << duration.count() << "ms" << std::endl;
    } else if (timeFrame == "s") {
        std::cout << message << duration.count() / 1000 << "s" << std::endl;
    } else {
        std::cout << "Unknown timeframe for " << message << std::endl;
    }
}

// scoped Timer features
Timer::Timer(std::string m, std::string tf) :
    message(m), timeFrame(tf) {

        start();
    }

Timer::~Timer() {
    stop();
    print();
}

// ==================== Connectivity Functions =======================

void assignMFtoGL(int* MFtoGL_GL, int numGL, int numMF, int MFmaxNumPresyn) {
    Timer timer("MF to GL: ", "ms");
    // first, ensure that each MF does not need to make more than 2 synapses
    // to occupy each GL
    assert((numMF >= ceil(numGL/MFmaxNumPresyn)) && "[Config Error] MF to GL connection not possible given the current numbers of cells and MF divergence.");
    // set array values all to -1 as a null indicator
    for (int i=0; i<numGL; i++)
        MFtoGL_GL[i] = -1;
    // ensure each MF connects to one GL
    // for each MF, chose a random GL from a shuffled list of them
    // ensures no doubles
    int* shuffled_GL = make_shuffled_arr(numGL);
    for (int i=0; i<numMF; i++) 
        MFtoGL_GL[shuffled_GL[i]] = i;
    delete[] shuffled_GL;

    // then, assign all the other GL a random MF
    // ensure each MF hasnt been used more than MFmaxNumPresyn times, using a counter array
    // this is not the fastest implementation, but it is simple and that is good.
    int* MF_synapses = new int[numMF];
    for (int i=0; i<numMF; i++) MF_synapses[i] = 1;

    auto rand_MF = make_generator(0, numMF);
    for (int i=0; i<numGL; i++) {
        if (MFtoGL_GL[i] == -1) {
            bool done = false;
            while (!done) {
                int mf = rand_MF();
                if (MF_synapses[mf] < MFmaxNumPresyn) {
                    MFtoGL_GL[i] = mf;
                    MF_synapses[mf]++;
                    done = true;
                }
            }
        }
    }
}

void connectGLtoGR(int** MFtoGR_GR, const int* MFtoGL_GL, int GRConv, int numGR,
    int GRx, int GRy, int GLspanX, int GLspanY, int GLx, int GLy) {
    Timer timer("GL to GR: ", "s");

    float scaleFactorX = static_cast<float>(GLx) / GRx;
    float scaleFactorY = static_cast<float>(GLy) / GRy;
    int divergenceCap = ((numGR * GRConv) / (GLx * GLy)) * 1.5;

    int* numGL_presyn = new int[GLx * GLy];
    std::memset(numGL_presyn, 0, sizeof(int) * GLx * GLy);
    std::vector<int> GL_availPre(GLx*GLy);
    std::iota(GL_availPre.begin(), GL_availPre.end(), 0);
    // make vector for all filtered results
    std::vector<int> candidate_GL;
    candidate_GL.reserve(GLspanX * GLspanY);       
    // need to store the pres that were chosen so they can be checked for exceeding
    // divergence limits afterword
    std::vector<int> chosen_pres;
    chosen_pres.reserve(GRConv);

    // need to set up a random device here because the range will change dynamically
    std::random_device rd;
    std::mt19937 gen(rd());   
    
    int* rand_GRs = make_shuffled_arr(numGR);
    // iterate through all GR cells
    for (int i=0; i<numGR; i++) {
        int GR_post = rand_GRs[i];
        int posX = round(scaleFactorX * (GR_post % GRx));
        int posY = round(scaleFactorY * (GR_post / GRx));

        // get vector of all presyn in span
        std::vector<int> GL_inSpan = make_candidate_vector_from_2D_range(posX, posY, GLspanX, GLspanY, GLx, GLy);
        std::sort(GL_inSpan.begin(), GL_inSpan.end());

        std::set_intersection(GL_inSpan.begin(), GL_inSpan.end(),
                              GL_availPre.begin(), GL_availPre.end(),
                              std::back_inserter(candidate_GL));
        // this is important to make sure that the possible connect span is large enough
        // to satisfy the number of connections needed to be made
        assert(candidate_GL.size() > GRConv);


        for (int j=0; j<GRConv; j++) {
            
            // chose a random index from the candidate array
            std::uniform_int_distribution<int> dist(0, candidate_GL.size() - 1);
            int rand_idx = dist(gen); 
            int GL_pre = candidate_GL[rand_idx];

            MFtoGR_GR[GR_post][j] = MFtoGL_GL[GL_pre];

            // remove that option from the vector of candidates, increment presyn, repeat
            candidate_GL.erase(candidate_GL.begin() + rand_idx);
            numGL_presyn[GL_pre]++;
            chosen_pres.emplace_back(GL_pre);
        }
        // from availPre, remove all GL that reached divergence threshold
        // only need to check this at the end of each post cell because they have no overlapping pre within
        for (int& pre : chosen_pres) {
            if (numGL_presyn[pre] >= divergenceCap) {
                // get the index value for that cell
                auto it = std::lower_bound(GL_availPre.begin(), GL_availPre.end(), pre);
                if (it != GL_availPre.end() && *it == pre) {
                    int idx = std::distance(GL_availPre.begin(), it);
                    // remove it from the candidate array
                    GL_availPre.erase(GL_availPre.begin() + idx);
                }
            }
        }
        // clear the arrays that are unique for each cell
        // this approach is better than removing the array and allocating again each cell
        candidate_GL.erase(candidate_GL.begin(), candidate_GL.end());
        chosen_pres.erase(chosen_pres.begin(), chosen_pres.end());
    }
    delete[] numGL_presyn;
    delete[] rand_GRs;
}

void connectGRtoGO(int** GRtoGO_GO, int GOConv, int numGO,
                   int GOx, int GOy, int GRx, int GRy,
                   int spanX, int spanY) {
    Timer timer("GR to GO: ", "ms");

    float scaleFactorX = static_cast<float>(GRx) / GOx;
    float scaleFactorY = static_cast<float>(GRy) / GOy;
    int divergenceCap = ((numGO * GOConv) / (GRx * GRy)) * 1.5;

    int* numGR_presyn = new int[GRx * GRy];
    std::memset(numGR_presyn, 0, sizeof(int) * GRx * GRy);
    std::vector<int> GR_availPre(GRx*GRy);
    std::iota(GR_availPre.begin(), GR_availPre.end(), 0);

    std::vector<int> candidate_GR;
    candidate_GR.reserve(spanX * spanY);
    std::vector<int> chosen_pres;
    chosen_pres.reserve(GOConv);

    std::random_device rd;
    std::mt19937 gen(rd());

    int* rand_GOs = make_shuffled_arr(numGO);
    for (int i=0; i<numGO; i++) {
        int GO_post = rand_GOs[i];
        int posX = round(scaleFactorX * (GO_post % GOx));
        int posY = round(scaleFactorY * (GO_post / GOx));

        std::vector<int> GR_inSpan = make_candidate_vector_from_2D_range(posX, posY, spanX, spanY, GRx, GRy);
        std::sort(GR_inSpan.begin(), GR_inSpan.end());

        std::set_intersection(GR_inSpan.begin(), GR_inSpan.end(),
                              GR_availPre.begin(), GR_availPre.end(),
                              std::back_inserter(candidate_GR));

        assert(candidate_GR.size() >= GOConv);

        for (int j=0; j<GOConv; j++) {
            assert(candidate_GR.size() > 0);

            std::uniform_int_distribution<int>dist(0, candidate_GR.size()-1);
            int rand_idx = dist(gen);
            int GR_pre = candidate_GR[rand_idx];

            GRtoGO_GO[i][j] = GR_pre;

            candidate_GR.erase(candidate_GR.begin() + rand_idx);
            numGR_presyn[GR_pre]++;
            chosen_pres.emplace_back(GR_pre);
        }

        for (int& pre : chosen_pres) {
            if (numGR_presyn[pre] >= divergenceCap) {
                auto it = std::lower_bound(GR_availPre.begin(), GR_availPre.end(), pre);
                if (it != GR_availPre.end() && *it == pre) {
                    int idx = std::distance(GR_availPre.begin(), it);
                    GR_availPre.erase(GR_availPre.begin() + idx);
                }
            }
        }
        candidate_GR.erase(candidate_GR.begin(), candidate_GR.end());
        chosen_pres.erase(chosen_pres.begin(), chosen_pres.end());
    }
    delete[] numGR_presyn;
    delete[] rand_GOs; 
}

void connectGOtoGR(int** GOtoGR_GR, int GRConv, 
                   int numGR, int spanX, int spanY,
                   int GOx, int GOy, int GRx, int GRy) {
    Timer timer("GO to GR: ", "ms");

    float scaleFactorX = static_cast<float>(GOx) / GRx;
    float scaleFactorY = static_cast<float>(GOy) / GRy;
    int divergenceCap = ((numGR * GRConv) / (GOx * GOy)) * 1.5;

    int* numGO_presyn = new int[GOx * GOy];
    std::memset(numGO_presyn, 0, sizeof(int) * GOx * GOy);
    std::vector<int> GO_availPre(GOx*GOy);
    std::iota(GO_availPre.begin(), GO_availPre.end(), 0);

    std::vector<int> candidate_GO;
    candidate_GO.reserve(spanX * spanY);
    std::vector<int> chosen_pres;
    chosen_pres.reserve(GRConv);

    std::random_device rd;
    std::mt19937 gen(rd());

    int* rand_GRs = make_shuffled_arr(numGR);
    for (int i=0; i<numGR; i++) {
        int GR_post = rand_GRs[i];
        int posX = round(scaleFactorX * (GR_post % GRx));
        int posY = round(scaleFactorY * (GR_post / GRx));

        std::vector<int> GO_inSpan = make_candidate_vector_from_2D_range(posX, posY, spanX, spanY, GOx, GOy);
        std::sort(GO_inSpan.begin(), GO_inSpan.end());

        std::set_intersection(GO_inSpan.begin(), GO_inSpan.end(),
                              GO_availPre.begin(), GO_availPre.end(),
                              std::back_inserter(candidate_GO));

        assert(candidate_GO.size() >= GRConv);


        for (int j=0; j<GRConv; j++) {

            std::uniform_int_distribution<int> dist(0, candidate_GO.size()-1);
            int rand_idx = dist(gen);
            int GO_pre = candidate_GO[rand_idx];
            
            GOtoGR_GR[GR_post][j] = GO_pre;

            // remove GO from candidates vector
            candidate_GO.erase(candidate_GO.begin() + rand_idx);
            numGO_presyn[GO_pre]++;
            chosen_pres.emplace_back(GO_pre);
        }

        for (int& pre : chosen_pres) {
            if (numGO_presyn[pre] >= divergenceCap) {
                auto it = std::lower_bound(GO_availPre.begin(), GO_availPre.end(), pre);
                if (it != GO_availPre.end() && *it == pre) {
                    int idx = std::distance(GO_availPre.begin(), it);
                    GO_availPre.erase(GO_availPre.begin() + idx);
                }
            }
        }
        candidate_GO.erase(candidate_GO.begin(), candidate_GO.end());
        chosen_pres.erase(chosen_pres.begin(), chosen_pres.end());
    }
    delete[] numGO_presyn;
    delete[] rand_GRs;
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