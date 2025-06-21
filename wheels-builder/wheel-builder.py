#!/usr/bin/env python3
"""
Mockup wheel builder script for cog validation
"""

from cog import BasePredictor, BaseModel, Input, Path

class Output(BaseModel):
    pass

class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(self) -> Output:
        print("Mockup wheel builder - this is just for cog validation")
        return Output()

if __name__ == "__main__":
    Predictor.predict() 