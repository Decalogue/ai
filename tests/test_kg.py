#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from unittest import TestCase, main
from ai.kg import KG

class TestMe(TestCase):
    def setUp(self):
        self.kg = KG(password="train")

    def test_merge(self):
        self.kg.merge(filepath="./sample/new.xls")

    def test_delete(self):
        pass

    def test_reset(self):
        # self.kg.reset()
        pass



if __name__ == '__main__':
    main()
