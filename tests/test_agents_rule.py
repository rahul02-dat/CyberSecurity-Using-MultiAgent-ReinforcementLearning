import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.agents.detection_rule import RuleBasedDetectionAgent
from src.agents.response_rule import RuleBasedResponseAgent


def test_detection_agent_no_attack():
    agent = RuleBasedDetectionAgent()
    obs = np.array([50.0, 600.0, 2.0, 5.0], dtype=np.float32)
    
    action = agent.act(obs)
    
    assert "flag" in action
    assert "confidence" in action
    assert action["flag"] == False
    assert action["confidence"] < 0.5


def test_detection_agent_high_flows():
    agent = RuleBasedDetectionAgent()
    obs = np.array([200.0, 600.0, 2.0, 5.0], dtype=np.float32)
    
    action = agent.act(obs)
    
    assert action["flag"] == False
    assert action["confidence"] >= 0.4


def test_detection_agent_bruteforce():
    agent = RuleBasedDetectionAgent()
    obs = np.array([50.0, 600.0, 25.0, 5.0], dtype=np.float32)
    
    action = agent.act(obs)
    
    assert action["flag"] == False
    assert action["confidence"] >= 0.3


def test_detection_agent_multiple_indicators():
    agent = RuleBasedDetectionAgent()
    obs = np.array([200.0, 300.0, 15.0, 7.0], dtype=np.float32)
    
    action = agent.act(obs)
    
    assert action["flag"] == True
    assert action["confidence"] >= 0.8


def test_response_agent_no_detection():
    agent = RuleBasedResponseAgent()
    obs = np.array([50.0, 600.0, 2.0, 5.0], dtype=np.float32)
    detection_result = {"flag": False, "confidence": 0.2}
    
    action = agent.act(obs, detection_result)
    
    assert action["action"] == "ALLOW"
    assert action["duration_bin"] == 0


def test_response_agent_low_confidence():
    agent = RuleBasedResponseAgent()
    obs = np.array([100.0, 600.0, 2.0, 5.0], dtype=np.float32)
    detection_result = {"flag": True, "confidence": 0.5}
    
    action = agent.act(obs, detection_result)
    
    assert action["action"] == "ALLOW"


def test_response_agent_medium_confidence():
    agent = RuleBasedResponseAgent()
    obs = np.array([150.0, 600.0, 2.0, 6.0], dtype=np.float32)
    detection_result = {"flag": True, "confidence": 0.7}
    
    action = agent.act(obs, detection_result)
    
    assert action["action"] == "QUARANTINE"
    assert action["duration_bin"] == 1


def test_response_agent_high_confidence():
    agent = RuleBasedResponseAgent()
    obs = np.array([300.0, 300.0, 15.0, 7.0], dtype=np.float32)
    detection_result = {"flag": True, "confidence": 0.9}
    
    action = agent.act(obs, detection_result)
    
    assert action["action"] == "BLOCK"
    assert action["duration_bin"] == 2


def test_agents_integration():
    detection_agent = RuleBasedDetectionAgent()
    response_agent = RuleBasedResponseAgent()
    
    obs = np.array([250.0, 350.0, 20.0, 6.5], dtype=np.float32)
    
    detection_action = detection_agent.act(obs)
    response_action = response_agent.act(obs, detection_action)
    
    assert detection_action["flag"] == True
    assert response_action["action"] in ["ALLOW", "QUARANTINE", "BLOCK"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])