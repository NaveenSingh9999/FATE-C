"""Base model class."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.compiled = False
        self.optimizer = None
        self.loss_fn = None
    
    @abstractmethod
    def forward(self, x):
        """Forward pass through the model."""
        pass
    
    def __call__(self, x):
        """Call the model on input x."""
        return self.forward(x)
    
    def compile(self, optimizer='adam', loss='auto'):
        """Compile the model."""
        from ..optim.adam import Adam
        from ..training.losses import auto_loss
        
        if optimizer == 'adam':
            self.optimizer = Adam()
        
        if loss == 'auto':
            self.loss_fn = auto_loss
        else:
            self.loss_fn = loss
            
        self.compiled = True
    
    def fit(self, x, y, epochs=1, batch_size=32, val=None):
        """Train the model."""
        if not self.compiled:
            self.compile()
        
        from ..training.trainer import Trainer
        trainer = Trainer(self, self.optimizer, self.loss_fn)
        return trainer.fit(x, y, epochs, batch_size, val)
    
    def predict(self, x):
        """Make predictions on input data."""
        return self.forward(x)
    
    def get_parameters(self):
        """Get all trainable parameters."""
        return {}
