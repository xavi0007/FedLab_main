# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .client.trainer import SerialClientTrainer
from .server.handler import ServerHandler
import matplotlib.pyplot as plt
import numpy as np

class StandalonePipeline(object):
    def __init__(self, handler: ServerHandler, trainer: SerialClientTrainer):
        """Perform standalone simulation process.

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer
        self.sampled_clients = None
        # initialization
        self.handler.num_clients = self.trainer.num_clients
        self.loss = []
        self.acc = []

    def get_sample_clients(self):
        return self.sampled_clients

    def main(self):
        while self.handler.if_stop is False:
            # server side
            self.sampled_clients = self.handler.sample_clients()
            # print(self.sampled_clients)
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, self.sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()
            # self.handler.evaluate()

    def evaluate(self):
        loss_, acc_ = self.handler.evaluate()
        self.loss.append(loss_)
        self.acc.append(acc_)

    def show(self):
        plt.figure(figsize=(8,4.5))
        ax = plt.subplot(1,2,1)
        # ax.plot(np.arange(len(self.loss)), self.loss)
        # ax.set_xlabel("Communication Round")
        # ax.set_ylabel("Loss")
        
        # ax2 = plt.subplot(1,2,2)
        fig, ax2 = plt.subplots(dpi = 300)
        ax2.plot(np.arange(len(self.acc)), self.acc)
        ax2.set_xlabel("Communication Round, T", fontsize=15,  fontweight='bold')
        ax2.set_ylabel('Performance', fontsize=15, fontweight='bold')

        ax2.legend(loc=1, fontsize= 15)
        plt.grid()
        plt.savefig("util.pdf", dpi = 300)
