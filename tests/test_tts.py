#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from unittest import TestCase, main
from ai.tts import TTS, LTTS

class TestMe(TestCase):
    def setUp(self):
        self.tts = LTTS()

    def test_say(self):
        self.tts.say("ok")

if __name__ == '__main__':
    main()
