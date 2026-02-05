from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict


class MessageBus:
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_history: List[Dict[str, Any]] = []
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        
        self.subscribers[topic].append(callback)
    
    def unsubscribe(self, topic: str, callback: Callable) -> None:
        
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)
    
    def publish(self, topic: str, message: Any, sender: str = "unknown") -> None:
        
        self.message_history.append({
            'topic': topic,
            'message': message,
            'sender': sender
        })
        
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)
    
    def get_subscribers(self, topic: str) -> List[Callable]:
        
        return self.subscribers.get(topic, [])
    
    def get_message_history(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        
        if topic is None:
            return self.message_history
        
        return [msg for msg in self.message_history if msg['topic'] == topic]
    
    def clear_history(self) -> None:
        
        self.message_history = []

class Message:
    
    def __init__(
        self,
        sender: str,
        recipient: str,
        content: Any,
        message_type: str = "default",
        priority: int = 0
    ):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type
        self.priority = priority
    
    def __repr__(self) -> str:
        return f"Message(from={self.sender}, to={self.recipient}, type={self.message_type})"