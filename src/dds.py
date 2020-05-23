"""See: https://github.com/dds-bridge/dds/blob/develop/doc/dll-description.md"""

from collections import defaultdict
from ctypes import Structure, c_int, pointer
import os
import platform

if platform.system() == "Windows":
    from ctypes import windll as libloader
else:
    from ctypes import cdll as libloader

DIRECTIONS = "NESW"
SUITS = "SHDC"
STRAINS = SUITS + "N"
RANKS = "??23456789TJQKA"
MAXNOOFBOARDS = 200

class Deal(Structure):
    _fields_ = [
        ("trump", c_int),
        ("first", c_int),
        ("currentTrickSuit", c_int * 3),
        ("currentTrickRank", c_int * 3),
        ("remainCards", (c_int * 4) * 4)
    ]

class FutureTricks(Structure):
    _fields_ = [
        ("nodes", c_int),
        ("cards", c_int),
        ("suit", c_int * 13),
        ("rank", c_int * 13),
        ("equals", c_int * 13),
        ("score", c_int * 13)
    ]

class DDTableDeal(Structure):
    _fields_ = [
        ("cards", (c_int * 4) * 4)
    ]

class DDTableResults(Structure):
    _fields_ = [
        ("resTable", (c_int * 4) * 5)
    ]

class Boards(Structure):
    _fields_ = [
        ("noOfBoards", c_int),
        ("deals", Deal * MAXNOOFBOARDS),
        ("target", c_int * MAXNOOFBOARDS),
        ("solutions", c_int * MAXNOOFBOARDS),
        ("mode", c_int * MAXNOOFBOARDS)
    ]

class SolvedBoards(Structure):
    _fields_ = [
        ("noOfBoards", c_int),
        ("solvedBoard", FutureTricks * MAXNOOFBOARDS)
    ]

class DDSError(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return f"Error code {self.code}"

def encode_deal(hands):
    cards = ((c_int * 4) * 4)()
    for i, direction in enumerate(DIRECTIONS):
        for card in hands[direction]:
            suit = SUITS.index(card[0])
            rank = RANKS.index(card[1])
            cards[i][suit] |= 1 << rank
    return cards

class DDS:
    def __init__(self, max_threads=0):
        if platform.system() == "Windows":
            libname = "dds.dll"
        else:
            libname = "libdds.so"
        self.libdds = libloader.LoadLibrary(libname)
        self.libdds.SetMaxThreads(max_threads)

    def solve_board(self, trump, first, current_trick, hands,
                    target=-1, solutions=3, mode=1, threadIndex=0):
        trump = STRAINS.index(trump)
        first = DIRECTIONS.index(first)

        currentTrickSuit = (c_int * 3)()
        currentTrickRank = (c_int * 3)()
        for i, card in enumerate(current_trick):
            currentTrickSuit[i] = SUITS.index(card[0])
            currentTrickRank[i] = RANKS.index(card[1])

        remainCards = encode_deal(hands)
        dl = Deal(trump, first, currentTrickSuit, currentTrickRank, remainCards)
        fut = FutureTricks()

        code = self.libdds.SolveBoard(dl, target, solutions, mode, pointer(fut), threadIndex)
        if code != 1:
            raise DDSError(code)

        scores = []
        for i in range(fut.cards):
            card = SUITS[fut.suit[i]] + RANKS[fut.rank[i]]
            score = fut.score[i]
            scores.append((card, score))
        return scores

    def calc_dd_table(self, hands):
        cards = encode_deal(hands)
        tableDeal = DDTableDeal(cards)
        table = DDTableResults()

        code = self.libdds.CalcDDtable(tableDeal, pointer(table))
        if code != 1:
            raise DDSError(code)

        results = dict()
        for strain, row in zip(STRAINS, table.resTable):
            results[strain] = dict(zip(DIRECTIONS, row))
        return results

    def solve_batch(self, trump, first, current_trick, batch,
                    target=-1, solutions=3, mode=1):
        trump = STRAINS.index(trump)
        first = DIRECTIONS.index(first)

        currentTrickSuit = (c_int * 3)()
        currentTrickRank = (c_int * 3)()
        for i, card in enumerate(current_trick):
            currentTrickSuit[i] = SUITS.index(card[0])
            currentTrickRank[i] = RANKS.index(card[1])

        results = defaultdict(list)

        nbuckets = (len(batch) + MAXNOOFBOARDS - 1) // MAXNOOFBOARDS
        for bucket in range(nbuckets):
            start = bucket * MAXNOOFBOARDS
            end = start + MAXNOOFBOARDS
            subbatch = batch[start:end]
            bo = Boards(noOfBoards=len(subbatch))

            for i, hands in enumerate(subbatch):
                remainCards = encode_deal(hands)
                bo.deals[i] = Deal(trump, first, currentTrickSuit, currentTrickRank, remainCards)
                bo.target[i] = target
                bo.solutions[i] = solutions
                bo.mode[i] = mode

            solved = SolvedBoards()

            code = self.libdds.SolveAllChunksBin(pointer(bo), pointer(solved), 1)
            if code != 1:
                raise DDSError(code)

            for fut in solved.solvedBoard[:solved.noOfBoards]:
                for i in range(fut.cards):
                    card = SUITS[fut.suit[i]] + RANKS[fut.rank[i]]
                    score = fut.score[i]
                    results[card].append(score)

        return results