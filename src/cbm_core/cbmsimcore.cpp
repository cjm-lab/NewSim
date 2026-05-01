/*
 * cbmsimcore.cpp
 *
 *  Created on: Dec 15, 2011
 *      Author: consciousness
 */

#include "cbmsimcore.h"
#include "logger.h"

//#define NO_ASYNC
//#define DISP_CUDA_ERR

CBMSimCore::CBMSimCore() {}

CBMSimCore::CBMSimCore(CBMState *state, int gpuIndStart, int numGPUP2) {
  CRandomSFMT0 randGen(time(0));
  int *mzoneRSeed = new int[state->getNumZones()];

  for (int i = 0; i < state->getNumZones(); i++) {
    mzoneRSeed[i] = randGen.IRandom(0, INT_MAX);
  }

  construct(state, mzoneRSeed, gpuIndStart, numGPUP2);

  delete[] mzoneRSeed;
}

CBMSimCore::~CBMSimCore() {
  for (int i = 0; i < numZones; i++) {
    delete zones[i];
  }

  delete[] zones;
  delete inputNet;

  for (int i = 0; i < numGPUs; i++) {
    // How could gpuIndStart ever not be 0,
    // given we're looping from 0 to numGPUs?
    cudaSetDevice(i + gpuIndStart);

    for (int j = 0; j < 8; j++) {
      cudaStreamDestroy(streams[i][j]);
    }
    delete[] streams[i];
  }

  delete[] streams;
}

void CBMSimCore::writeToState() {
  inputNet->writeToState();

  for (int i = 0; i < numZones; i++) {
    zones[i]->writeToState();
  }
}

void CBMSimCore::writeState(std::fstream &outfile) {
  writeToState();
  simState->writeState(outfile); // using internal cp
}

void CBMSimCore::initCUDAStreams() {
  cudaError_t error;

  int maxNumGPUs;
  error = cudaGetDeviceCount(&maxNumGPUs);

  LOG_DEBUG("CUDA max num devices: %d", maxNumGPUs);
  LOG_DEBUG("%s", cudaGetErrorString(error));
  LOG_DEBUG("CUDA num devices: %d, starting at GPU %d", numGPUs, gpuIndStart);

  streams = new cudaStream_t *[numGPUs];

  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    LOG_DEBUG("Selecting device %d", i);
    LOG_DEBUG("%s", cudaGetErrorString(error));
    streams[i] = new cudaStream_t[8];
    LOG_DEBUG("Resetting device %d", i);
    LOG_DEBUG("%s", cudaGetErrorString(error));
    cudaDeviceSynchronize();

    for (int j = 0; j < 8; j++) {
      error = cudaStreamCreate(&streams[i][j]);
      LOG_DEBUG("Initializing stream %d for device %d", j, i);
      LOG_DEBUG("%s", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    LOG_DEBUG("Cuda device %d", i);
    LOG_DEBUG("%s", cudaGetErrorString(error));
  }
}

void CBMSimCore::initAuxVars() { curTime = 0; }

void CBMSimCore::syncCUDA(std::string title) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
#ifdef DISP_CUDA_ERR
    LOG_TRACE("sync point  %s, switching to gpu %d", title.c_str(), i);
    LOG_TRACE("%s", cudaGetErrorString(error));
#endif
    error = cudaDeviceSynchronize();
#ifdef DISP_CUDA_ERR
    LOG_TRACE("sync point  %s, switching to gpu %d", title.c_str(), i);
    LOG_TRACE("%s", cudaGetErrorString(error));
#endif
  }
}
//**************************************************************
void CBMSimCore::calcActivity(float spillFrac, enum plasticity pf_pc_plast,
                              enum plasticity mf_nc_plast, uint32_t use_cs,
                              uint32_t use_us, bool stp_on) {
  syncCUDA("1");

  curTime++;

  // cp mf spike activity to gpu
  inputNet->cpyAPMFHosttoGPUCUDA(streams, 6);
  // update mf -> gr synaptic variables
  nvtxRangePushA("UpdateMFtoGROut");
  inputNet->updateMFtoGROut();
  nvtxRangePop();
  // cpy mf -> gr depression amplitude to gpu
  inputNet->cpyDepAmpMFHosttoGPUCUDA(streams, 5);

  // update gr inputs from mf
  inputNet->runUpdateMFInGRDepressionCUDA(streams, 2);
  inputNet->runUpdateMFInGRCUDA(streams, 0);

  // the gr spiking activity kernel
  inputNet->runGRActivitiesCUDA(streams, 0);

  // update gr spike history array
  inputNet->runUpdateGRHistoryCUDA(streams, 4, curTime);

  // update the output variables for gr -> go synapse
  //inputNet->runUpdateGROutGOCUDA(streams, 7);
  // sum over go inputs from gr
  // inputNet->runSumGRGOOutCUDA(streams, 4);
  // NEW GR->GO implementation
  inputNet->runUpdateSumGRGOOutCUDA(streams, 3);
  // cpy resulting sums from device to host
  inputNet->cpyGRGOSumGPUtoHostCUDA(streams, 3);

  // update mf output to go for mf -> go synapse
  nvtxRangePushA("updateMFtoGO");
  inputNet->updateMFtoGOOut();
  nvtxRangePop();
  // golgi spiking activity function (on host)
  nvtxRangePushA("calcGOActivites");
  inputNet->calcGOActivities();
  nvtxRangePop();

  // update go <-> go output params
  nvtxRangePushA("updateGOtoGO");
  inputNet->updateGOtoGOOut();
  nvtxRangePop();
  // update go -> gr output params
  nvtxRangePushA("UpdateGOtoGRParameters");
  inputNet->updateGOtoGROutParameters(spillFrac);
  nvtxRangePop();

  // copy depression amplitude from go -> gr from host to device
  inputNet->cpyDepAmpGOGRHosttoGPUCUDA(
      streams, 2); // NOTE: currently does nothing (08/11/2022)
  // copy dynamic amplitude from host to device
  inputNet->cpyDynamicAmpGOGRHosttoGPUCUDA(streams, 3);
  // copy go spikes to device
  inputNet->cpyAPGOHosttoGPUCUDA(streams, 7);

  // syncCUDA("2");
  // run update input function go -> gr synapse
  inputNet->runUpdateGOInGRCUDA(streams, 1);
  // run update input for depression amplitude go -> gr
  inputNet->runUpdateGOInGRDepressionCUDA(streams, 3);
  // run dynamic spillover input go -> gr
  inputNet->runUpdateGOInGRDynamicSpillCUDA(streams, 4);

  // perform pf -> pc plasticity
  for (int i = 0; i < numZones; i++) {
    if (stp_on) {
      zones[i]->runPFPCSTPCUDA(streams, 0, use_cs, use_us);
    }
    if (pf_pc_plast == GRADED) {
      zones[i]->runPFPCGradedPlastCUDA(streams, 1, curTime);
    } else if (pf_pc_plast == BINARY) {
      zones[i]->runPFPCBinaryPlastCUDA(streams, 1, curTime);
    } else if (pf_pc_plast == ABBOTT_CASCADE) {
      zones[i]->runPFPCAbbottCascadePlastCUDA(streams, 1, curTime);
    } else if (pf_pc_plast == MAUK_CASCADE) {
      zones[i]->runPFPCMaukCascadePlastCUDA(streams, 1, curTime);
    }
  }

  /* mzone (stripe) computation */
  for (int i = 0; i < numZones; i++) {
    // update pf output on the device
    zones[i]->runPFPCOutCUDA(streams, i + 2);
    // update pfpc sums on device
    zones[i]->runPFPCSumCUDA(streams, 1);
    // copy pfpc sums to host
    zones[i]->cpyPFPCSumCUDA(streams, i + 2);

    // update input from pf to bc and sc
    zones[i]->runUpdatePFBCSCOutCUDA(
        streams, i + 4); // adding i might break things in future
    // run sum over bc input from pc on gpu
    zones[i]->runSumPFBCCUDA(streams, 2);
    // run sum over sc input from pc on gpu
    zones[i]->runSumPFSCCUDA(streams, 3);
    // cpy pf -> bc input sum to host
    zones[i]->cpyPFBCSumGPUtoHostCUDA(streams, 5);
    // cpy pf -> sc input sum to host
    zones[i]->cpyPFSCSumGPUtoHostCUDA(streams, 3);

    // calculate sc spiking activity (host)
    nvtxRangePushA("CalcSCActivities");
    zones[i]->calcSCActivities();
    nvtxRangePop();
    // calculate bc spiking activity (host)
    nvtxRangePushA("CalcBCActivities");
    zones[i]->calcBCActivities();
    nvtxRangePop();
    // update spike outputs from bc -> pc
    nvtxRangePushA("updateBCtoPC");
    zones[i]->updateBCPCOut();
    nvtxRangePop();
    // update spike outputs from sc -> pc
    nvtxRangePushA("updateSCtoPC");
    zones[i]->updateSCPCOut();
    nvtxRangePop();

    // compute pc spiking activity (host)
    nvtxRangePushA("calcPCActivities");
    zones[i]->calcPCActivities();
    nvtxRangePop();
    // update pc output vars
    nvtxRangePushA("UpdatePCOut");
    zones[i]->updatePCOut();
    nvtxRangePop();

    // compute io activities (host)
    
    nvtxRangePushA("calcIOActivities");
    zones[i]->calcIOActivities();
    nvtxRangePop();
    // update io output variables
    //std::cout << "before updateIO\n";
    nvtxRangePushA("updateIOOut");
    zones[i]->updateIOOut();
    nvtxRangePop();
    //std::cout << "after updateIO\n";
    // temp solution: by default mfnc plast is GRADED. no other
    // plasticity modes are given for these synapses
    if (mf_nc_plast != OFF) {
      nvtxRangePushA("UpdateMFNCSyn");
      zones[i]->updateMFNCSyn(inputNet->exportHistMF(), curTime);
      nvtxRangePop();
    }
    
    // update mf -> nc output vars
    nvtxRangePushA("UpdateMFNCOut");
    zones[i]->updateMFNCOut();
    nvtxRangePop();
    // compute nc spiking activity
    
    nvtxRangePushA("calcNCActivities");
    zones[i]->calcNCActivities();
    nvtxRangePop();
    // update nc output vars
   
    nvtxRangePushA("updateNCOut");
    zones[i]->updateNCOut();
    nvtxRangePop();
  }

  // reset mf histories, given the current time
  nvtxRangePushA("resetMFHist");
  inputNet->resetMFHist(curTime);
  nvtxRangePop();
  
}
// end of calcActivities

void CBMSimCore::updateMFInput(const uint8_t *mfIn) {
  nvtxRangePushA("updateMFActivities");
  inputNet->updateMFActivties(mfIn);
  nvtxRangePop();

  for (int i = 0; i < numZones; i++) {
    nvtxRangePushA("updateMFActivities");
    zones[i]->updateMFActivities(mfIn);
    nvtxRangePop();
  }
}

void CBMSimCore::setTrueMFs(const bool *isCollateralMF) {

  for (int i = 0; i < numZones; i++) {
    zones[i]->setTrueMFs(isCollateralMF);
  }
}

void CBMSimCore::updateGRStim(int startGRStim_, int numGRStim_) {
  isGRStim = true;
  this->numGRStim = numGRStim_;
  this->startGRStim = startGRStim_;
}

void CBMSimCore::updateErrDrive(unsigned int zoneN, float errDriveRelative) {
  zones[zoneN]->setErrDrive(errDriveRelative);
}

InNet *CBMSimCore::getInputNet() { return (InNet *)inputNet; }

MZone **CBMSimCore::getMZoneList() { return (MZone **)zones; }

void CBMSimCore::construct(CBMState *state, int *mzoneRSeed, int gpuIndStart,
                           int numGPUP2) {
  int maxNumGPUs;

  numZones = state->getNumZones();
  
  cudaGetDeviceCount(&maxNumGPUs);

  if (gpuIndStart <= 0) {
    this->gpuIndStart = 0;
  } else if (gpuIndStart >= maxNumGPUs) {
    this->gpuIndStart = maxNumGPUs - 1;
  } else {
    this->gpuIndStart = gpuIndStart;
  }

  if (numGPUP2 < 0) {
    numGPUs = maxNumGPUs;
  } else {
    numGPUs = (unsigned int)numGPUP2;
  }

  if (this->gpuIndStart + numGPUs > maxNumGPUs) {
    numGPUs = 1;
  }
  LOG_DEBUG("Calculated (?) number of GPUs: %d", numGPUs);

  LOG_DEBUG("Initializing cuda streams...");
  
  initCUDAStreams();
  LOG_DEBUG("Finished initialzing cuda streams.");

  // NOTE: inputNet has internal cp, no need to pass to constructor
  inputNet =
      new InNet(state->getInnetConStateInternal(),
                state->getInnetActStateInternal(), this->gpuIndStart, numGPUs);

  zones = new MZone *[numZones];

  for (int i = 0; i < numZones; i++) {
    // same thing for zones as with innet
    zones[i] =
        new MZone(streams, state->getMZoneConStateInternal(i),
                  state->getMZoneActStateInternal(i), mzoneRSeed[i],
                  inputNet->getApBufGRGPUPointer(),
                  inputNet->getHistGRGPUPointer(), this->gpuIndStart, numGPUs);
  }
  LOG_DEBUG("Mzone construction complete");
  initAuxVars();
  LOG_DEBUG("AuxVars good");

  simState = state; // shallow copy
}
