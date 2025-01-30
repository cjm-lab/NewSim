/*
 * ecmfpopulation.cpp
 *
 *  Created on: Jul 13, 2014
 *      Author: consciousness
 */

#include <random>

#include "ecmfpopulation.h"
#include "file_utility.h"
#include "logger.h"
// Changed MF background firing rates 1.0 to 55.0  I change them back to 10 and 30

ECMFPopulation::ECMFPopulation() {

  /* initialize mf frequency population variables */
  CRandomSFMT0 ISIGen(randSeed);
  LOG_DEBUG("Allocating mf memory...");
  allocateMemory();
  LOG_DEBUG("Finished allocating mf memory.");

  int numCS = fracCS * num_mf;
  int numColl = fracColl * num_mf;
  u_int32_t MFindex;
  for (u_int32_t isi = 0;isi<num_mf;isi++){
    MFisi[isi] = ISIGen.IRandom(1, 10); // have to start with non-zero inter-spike intervals
    // std::cout << MFisi[isi] << "\n";
  }
// MFisiDistribution holds ISIs to generate MF activity at Frequencies between 1 and 100 Hz
  std::random_device rd;
  std::mt19937 gen(rd());
  CRandomSFMT0 randGen(randSeed);
 for (u_int32_t i=0;i<101;i++){
    double f = 1000.0/(i*1.0);
    double f_std = f/5;
     std::normal_distribution<double> dist(f,f_std);

    for (u_int32_t j=0;j<10000;j++){
      double temp = dist(gen);
      if (temp>100000){
        temp = 5;
      }
      if (temp < 5){
        temp = 5;
      }
      MFindex = (i*10000)+j;
      MFisiDistribution[MFindex]=static_cast<u_int32_t>(temp); 
      // std::cout << MFindex << "  " << MFisiDistribution[MFindex] << "  " << temp <<"\n";
      //std::cout << MFisiDistribution[MFindex] << ",";
    }
    //std::cout << "\n";
 }
  
  LOG_DEBUG("Setting CS Mfs...");
  // Pick MFs for CS
  setMFs(numCS, num_mf, randGen, isAny, isCS);
  LOG_DEBUG("Finished setting CS Mfs...");

  // Set the collaterals
  if (turnOnColls) {
    LOG_DEBUG("Setting Collateral Mfs...");
    setMFs(numColl, num_mf, randGen, isAny, isColl);
    LOG_DEBUG("Finished setting Collateral Mfs.");
  }
  LOG_DEBUG("Setting Mf frequencies...");
  // u_int16_t maxBG = 0;
  // u_int16_t minBG = 150;  
  // u_int16_t maxCS = 0;
  // u_int16_t minCS = 150; 
  for (u_int16_t i = 0; i < num_mf; i++) {
    // if (isColl[i]) {
    //   mfFreqBG[i] = -1;
    //   mfFreqCS[i] = -1;
    // } else {
        mfFreqBG[i] = randGen.IRandom(bgFreqMin, bgFreqMax);
        mfFreqCS[i] = mfFreqBG[i];
      //mfFreqBG[i] = static_cast<u_int16_t>(randGen.Random() * (bgFreqMax - bgFreqMin) + bgFreqMin);
      //mfFreqBG[i] *= sPerTS * kappa;
      if (isCS[i]) {
        mfFreqCS[i] = randGen.IRandom(csFreqMin, csFreqMax);
        
        //mfFreqCS[i] = static_cast<u_int16_t>(randGen.Random() * (csFreqMax - csFreqMin) + csFreqMin);
        //mfFreqCS[i] *= sPerTS * kappa;
      }
    // }
  }
 
  // }
  LOG_DEBUG("Finished setting Mf frequencies.");

  LOG_DEBUG("Preparing Collaterals...");
  prepCollaterals(randSeedGen->IRandom(0, INT_MAX));
  LOG_DEBUG("Finished preparing Collaterals...");
}

ECMFPopulation::ECMFPopulation(std::fstream &infile) {
  LOG_DEBUG("Allocating mf memory...");
  allocateMemory();
  LOG_DEBUG("Finished allocating mf memory.");
  LOG_DEBUG("Loading mfs from file...");
  rawBytesRW((char *)mfFreqBG, num_mf * sizeof(float), true, infile);
  rawBytesRW((char *)mfFreqCS, num_mf * sizeof(float), true, infile);
  rawBytesRW((char *)isCS, num_mf * sizeof(bool), true, infile);
  rawBytesRW((char *)isColl, num_mf * sizeof(bool), true, infile);
  rawBytesRW((char *)dnCellIndex, num_mf * sizeof(uint32_t), true, infile);
  rawBytesRW((char *)mZoneIndex, num_mf * sizeof(uint32_t), true, infile);
  LOG_DEBUG("finished loading mfs from file.");
  std::random_device rd;
  std::mt19937 gen(rd());
  CRandomSFMT0 ISIGen(randSeed);
  u_int32_t MFindex;

  for (u_int32_t isi = 0;isi<num_mf;isi++){
    MFisi[isi] = ISIGen.IRandom(1, 10); // have to start with non-zero inter-spike intervals
    // std::cout << MFisi[isi] << "\n";
  }
  CRandomSFMT0 randGen(randSeed);
  //std::ofstream outFile("ISIdist.txt");

  for (u_int32_t i=0;i<101;i++){
      double f = 1000.0/(i*1.0);
      //double f = 1000.0/(40);  // change this back mike
      double f_std = f/2;
      std::normal_distribution<double> dist(f,f_std);

      for (u_int32_t j=0;j<10000;j++){
        double temp = dist(gen);
        if (temp>100000){
          temp = 5;
        }
        if (temp < 5){
          temp = 5;
        }
        MFindex = (i*10000)+j;
        MFisiDistribution[MFindex]=static_cast<u_int32_t>(temp); 
        // std::cout << MFindex << "  " << MFisiDistribution[MFindex] << "  " << temp <<"\n";
        // outFile << MFisiDistribution[MFindex] << ",";  
      }
      // outFile << "\n";
  }
  // outFile.close();
}

ECMFPopulation::~ECMFPopulation() {
  delete[] mfFreqBG;
  delete[] mfFreqCS;
  delete[] MFisiDistribution;
  delete[] MFisi;
  delete[] isCS;
  delete[] isColl;
  delete[] isAny;

  delete randSeedGen;
  delete noiseRandGen;
  for (uint32_t i = 0; i < nThreads; i++) {
    delete randGens[i];
  }

  delete[] randGens;
  delete normDist;
  free(aps);
  free(apBufs);
  free(dnCellIndex);
  free(mZoneIndex);
}

/* public methods except constructor and destructor */
void ECMFPopulation::writeToFile(std::fstream &outfile) {
  rawBytesRW((char *)mfFreqBG, num_mf * sizeof(float), false, outfile);
  rawBytesRW((char *)mfFreqCS, num_mf * sizeof(float), false, outfile);
  rawBytesRW((char *)isCS, num_mf * sizeof(bool), false, outfile);
  rawBytesRW((char *)isColl, num_mf * sizeof(bool), false, outfile);
  rawBytesRW((char *)dnCellIndex, num_mf * sizeof(uint32_t), false, outfile);
  rawBytesRW((char *)mZoneIndex, num_mf * sizeof(uint32_t), false, outfile);
}

void ECMFPopulation::writeMFLabels(std::string labelFileName) {
  LOG_DEBUG("Writing MF labels...");
  std::fstream mflabels(labelFileName.c_str(), std::fstream::out);

  for (int i = 0; i < num_mf; i++) {
    if (isColl[i]) {
      mflabels << "col ";
    } else if (isCS) {
      mflabels << "ton ";
    } else {
      mflabels << "bac ";
    }
  }
  mflabels.close();
  LOG_DEBUG("MF labels written.");
}

const u_int16_t *ECMFPopulation::getBGFreq() { return mfFreqBG; }

const u_int16_t *ECMFPopulation::getCSFreq() { return mfFreqCS; }

const bool *ECMFPopulation::getCSIds() { return isCS; }

const bool *ECMFPopulation::getCollIds() { return isColl; }

void ECMFPopulation::calcGammaActivity(enum mf_type type, MZone **mZoneList, int change) {
  //CRandomSFMT0 nextGen((unsigned) time(NULL));

  u_int16_t *frequencies = (type == CS) ? mfFreqCS : mfFreqBG;
  u_int32_t MFindex2;
  //std::ofstream outFile("actualISIs.txt", std::ios::app);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0,9999);

  for (u_int32_t i = 0; i < num_mf; i++) {
    aps[i] = 0;
    if (change == 1){
      //MFindex2 = (frequencies[i]*10000)+nextGen.IRandom(0,9999);
      MFindex2 = (frequencies[i]*10000)+dist(gen);
      if (MFisiDistribution[MFindex2]<MFisi[i]) {
        MFisi[i] = MFisiDistribution[MFindex2];
      }
    }
      // std::cout << MFisi[i] << "\n";
    if (MFisi[i]<=1) {
      aps[i] = 1;
      MFindex2 = (frequencies[i]*10000)+dist(gen);
      // MFindex2 = (100*10000)+nextGen.IRandom(0,9999);  // specify fixed firing rate
      MFisi[i] = MFisiDistribution[MFindex2];
      // if (i==0){
      //   std::cout << frequencies[i] << " " << MFindex2 << " " << MFisiDistribution[MFindex2] << " " << MFisi[i] << "\n";
      
      //   // outFile <<  i << "," << frequencies[i] << "," << MFisi[i] << "," << MFindex2 << "\n";
      // }
    }
    else {
      MFisi[i] -= 1;
    }
  }
   //outFile.close();
}

const uint8_t *ECMFPopulation::getAPs() { return (const uint8_t *)aps; }

/* private methods */

void ECMFPopulation::allocateMemory() {
  mfFreqBG = new u_int16_t[num_mf]();
  mfFreqCS = new u_int16_t[num_mf]();
  MFisi = new u_int32_t[num_mf]();
  MFisiDistribution = new u_int32_t[101*10000]();

  isCS = new bool[num_mf]();
  isColl = new bool[num_mf]();
  isAny = new bool[num_mf]();

  /* initializing poisson gen vars */

  randSeedGen = new CRandomSFMT0(randSeed);
  randGens = new CRandomSFMT0 *[nThreads];

  for (uint32_t i = 0; i < nThreads; i++) {
    randGens[i] = new CRandomSFMT0(randSeedGen->IRandom(0, INT_MAX));
  }

  normDist = new std::normal_distribution<float>(0, this->noiseSigma);
  noiseRandGen = new std::mt19937(randSeed);

  aps = (uint8_t *)calloc(num_mf, sizeof(uint8_t));
  apBufs = (uint32_t *)calloc(num_mf, sizeof(uint32_t));

  dnCellIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));
  mZoneIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));
}

void ECMFPopulation::setMFs(int numTypeMF, int num_mf, CRandomSFMT0 &randGen,
                            bool *isAny, bool *isType) {
  for (int i = 0; i < numTypeMF; i++) {
    while (true) {
      int mfInd = randGen.IRandom(0, num_mf - 1);

      if (!isAny[mfInd]) {
        isAny[mfInd] = true;
        isType[mfInd] = true;
        break;
      }
    }
  }
}

void ECMFPopulation::prepCollaterals(int rSeed) {
  uint32_t repeats = num_mf / (numZones * num_nc) + 1;
  uint32_t *tempNCs = new uint32_t[repeats * numZones * num_nc];
  uint32_t *tempMZs = new uint32_t[repeats * numZones * num_nc];

  for (uint32_t i = 0; i < repeats; i++) {
    for (uint32_t j = 0; j < numZones; j++) {
      for (uint32_t k = 0; k < num_nc; k++) {
        tempNCs[k + num_nc * j + num_nc * numZones * i] = k;
        tempMZs[k + num_nc * j + num_nc * numZones * i] = j;
      }
    }
  }
  std::srand(rSeed);
  std::random_shuffle(tempNCs, tempNCs + repeats * numZones * num_nc);
  std::srand(rSeed);
  std::random_shuffle(tempMZs, tempMZs + repeats * numZones * num_nc);
  std::copy(tempNCs, tempNCs + num_mf, dnCellIndex);
  std::copy(tempMZs, tempMZs + num_mf, mZoneIndex);

  delete[] tempNCs;
  delete[] tempMZs;
}
