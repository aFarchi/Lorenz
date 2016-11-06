#! /usr/bin/env python

#__________________________________________________
# pyLorenz/truth/
# abstracttruth.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# abstract class to handle the truth
#

#__________________________________________________

class AbstractTruth(object):

    #_________________________

    def __init__(self, t_xSpaceDimension, t_ySpaceDimension, t_observationTimes, t_output, t_truthOutputFields):
        self.setAbstractTruthParameters(t_xSpaceDimension, t_ySpaceDimension, t_observationTimes, t_output, t_truthOutputFields)

    #_________________________

    def setAbstractTruthParameters(self, t_xSpaceDimension, t_ySpaceDimension, t_observationTimes, t_output, t_truthOutputFields):
        # observation times
        self.m_observationTimes  = t_observationTimes
        # output
        self.m_output            = t_output
        # truth output fields
        self.m_truthOutputFields = t_truthOutputFields
        # x, y space dimension
        self.m_xSpaceDimension   = t_xSpaceDimension
        self.m_ySpaceDimension   = t_ySpaceDimension
        # time
        self.m_time              = 0.0

    #_________________________

    def temporaryRecordShape(self, t_nRecord, t_field):
        # shape for temporary array recording the given field
        if t_field == 'trajectory':
            return (t_nRecord, self.m_xSpaceDimension)
        elif t_field == 'observations':
            return (t_nRecord, self.m_ySpaceDimension)
        elif t_field == 'time':
            return (t_nRecord,)
        return (t_nRecord, 0)

    #_________________________

    def observation(self):
        # access function for observation array
        raise NotImplementedError

    #_________________________

    def truth(self):
        # access function for truth array
        raise NotImplementedError

    #_________________________

    def initialise(self):
        # initialise output
        self.m_output.initialiseTruthOutput(self.m_truthOutputFields, self.temporaryRecordShape)

    #_________________________

    def forecast(self, t_tStart, t_tEnd):
        # integrate truth from tStart to tEnd
        raise NotImplementedError

    #_________________________

    def permute(self):
        # permute array to prepare next cycle
        raise NotImplementedError

    #_________________________

    def record(self):
        # trajectory
        self.m_output.record('truth', 'trajectory', self.truth())
        # observation
        self.m_output.record('truth', 'observations', self.observation())
        # time
        self.m_output.record('truth', 'time', self.m_time)

    #_________________________

    def finalise(self):
        # end simulation
        pass

#__________________________________________________

