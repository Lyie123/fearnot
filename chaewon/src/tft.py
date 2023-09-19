from time import sleep, time, perf_counter
from screenshot import FastShot
from pipe import PipeServer
from typing import Dict, List
from dataclasses import dataclass
from rich import print
from json import dumps
from pprint import pprint
from ultralytics import YOLO
from json import dumps
from rich import print

import requests as r
import subprocess
import pytesseract
import numpy as np
import cv2
import keyboard
import os

class Tft:
	def __init__(self, activate_overlay=False):
		self._screenshot = FastShot('League of Legends (TM) Client') #FastShot('Sologesang - Twitch - Google Chrome') 
		self._screenshot.search()
		self._wait_time_stage_transition = 3
		self._game_state = None
		self._overlay = activate_overlay
		self._api_endpoint = 'http://127.0.0.1:8000/predict/' 
		
		if activate_overlay:
			self._pipe = PipeServer('CSServer')
			subprocess.call(r'C:/Users/Rene/Desktop/projects/fearnot/chaewon/Chaewon/bin/Release/net7.0/Chaewon.exe')
			self.connect()


		self._hotkeys = {
			'deploy': 'W',
			'sell': 'E',
			'refresh': 'D',
			'buy_xp': 'F',
			'scout_clockwise': '1',
			'scout_counterclockwise': '3',
			'scout_place_1': 'alt+1',
			'scout_place_2': 'alt+2',
			'scout_place_3': 'alt+3',
			'scout_place_4': 'alt+4',
			'scout_place_5': 'alt+5',
			'scout_place_6': 'alt+6',
			'scout_place_7': 'alt+7',
			'scout_place_8': 'alt+8',
			'scout_next_place': ']',
			'scout_previous_place': '[',
			'center_on_tactician': '2',
			'toogle_recap': 'Tab',
			'toogle_opponent_recap': 'A',
			#'toggle_health_bars': 'N',
			'toggle_summoner_names': 'M',
			'toggle_hud': 'O',
			'focus_self': 'Space'
		}
	
	def _press_key(self, activity: str, delay: float=0.1):
		keyboard.send(activity)
		sleep(delay)

	def _extract(self, img: np.array) -> Dict:
		encoded = cv2.imencode('.jpg', img)[1].tobytes()
		result = r.post(self._api_endpoint, files={'image': encoded})
		return result.json()

	def scout_player(self, rank: int) -> np.ndarray:
		assert 0 < rank < 9, f'rank have to be between 1-8, got {rank}'
		self._press_key(f'alt+{rank}')
		return self._screenshot.capture()
	
	def scout_clockwise(self) -> np.ndarray:
		self._press_key(self._hotkeys['scout_clockwise'])
		return self._screenshot.capture()

	def toggle_hud(self) -> None:
		self._press_key(self._hotkeys['toggle_hud'])
		return

	def scout_all_players(self, order_by='clockwise', focus_afterwards: bool=True) -> List[np.ndarray]:
		self.toggle_hud()
		if order_by == 'clockwise':
			screenshots = [self.scout_clockwise() for _ in range(1, 9)]

		elif order_by == 'placewise':
			screenshots = [self.scout_player(r) for r in range(1, 9)]

		if focus_afterwards:
			self.focus_player()
		
		self.toggle_hud()

		return screenshots

	def focus_player(self) -> None:
		keyboard.send(self._hotkeys['focus_self'])
		return

	def get_gamestate(self) -> Dict:
		img = self._screenshot.capture(update_hwnd=False)
		buffer = self._extract(img)

		if self._game_state is not None:
			if buffer['stage'] > self._game_state['stage']:
				buffer['events'].append('stage_changed')

		self._game_state = buffer

		if self._overlay:
			self._pipe.write(dumps([]))
			sleep(0.1)
			self._pipe.write(dumps(self._game_state['bench-champion'] + self._game_state['board-champion']))

		return self._game_state

	def connect(self) -> None:
		self._pipe.connect()
		print('successfully connected')

	@property
	def is_connected(self):
		return True


if __name__ == '__main__':
	tft = Tft()
	while True:
		if keyboard.is_pressed(','):
			os.system('cls')
			state = tft.get_gamestate()
			print(state)
			sleep(1)

		if keyboard.is_pressed('.'):
			exit()