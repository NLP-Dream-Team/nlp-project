"""
A module for obtaining repo readme and language data from the github API.

Before using this module, read through it, and follow the instructions marked
TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['assaf/zombie',
 'IAIK/ZombieLoad',
 'marblexu/PythonPlantsVsZombies',
 'amazon-archives/aws-lambda-zombie-workshop',
 'ErLinErYi/PlantsVsZombies',
 'sivvig/ZombieBird',
 'loomnetwork/zombie-char-component',
 'codeschool/RFZ2-ZombieTweets',
 'JetBoom/zombiesurvival',
 'eliortabeka/zombie-mayhem',
 'tra38/ZombieWriter',
 'dsnezhkov/zombieant',
 'llx330441824/plant_vs_zombie_simple',
 'CompleteUnityDeveloper/09-ZombieRunner-Original',
 'Fankouzu/my-crypto-zombie',
 'sgtcaze/ZombieEscape',
 'adamtuliper/ZombiePumpkinSlayer',
 'Z6543/ZombieBrowserPack',
 'kangwang1988/XcodeZombieCode',
 'rico345100/unity-zombie-defence-fps-example',
 'L8426936/CleanUpWeChatZombieFans',
 'Dislaik/zombieoutbreak',
 'minkphp/MinkZombieDriver',
 'mrbarbasa/js-zombies',
 'Zhuagenborn/Plants-vs.-Zombies-Online-Battle',
 'JoakimCarlsson/ColdWarZombieTrainer',
 'simonmonk/zombies',
 'Alig96/nzombies',
 'Unity-Technologies/ZombieObjectDetector',
 'ryanpetrello/python-zombie',
 'handsomestone/Zombie-Defense-Game',
 'nexusnode/crafting-dead',
 'asweigart/zombiedice',
 'hoaproject/Zombie',
 'arminkz/PlantsVsZombies',
 'johnBuffer/ZombieV',
 'Plugily-Projects/Village_Defense',
 'Corosauce/ZombieAwareness',
 'tbl00c/PlantsVsZombies',
 'sahan/RoboZombie',
 'BenWirus/ZombieVoters',
 'zombie33g123ss/ZombieXSystemv1',
 'pardeike/Zombieland',
 '0xsha/ZombieVPN',
 'wohow/zombie',
 'Dislaik/esx_zombiesystem',
 'fisadev/zombsole',
 'qubka/Zombie-Plague',
 'nigeon/CryptoZombies',
 'interfaced/zombiebox',
 'fegemo/cefet-web-zombie-garden',
 'Kampfkarren/zombie-strike',
 'Vector35/PwnAdventureZ',
 'tunm123/PlantsVsZombies',
 'BhavyaC16/Plants-Vs-Zombies',
 'Z8pn/RageSurvival',
 'Jbleezy/BO1-Reimagined',
 'williamfiset/Survival',
 'mattbierbaum/zombies-usa',
 'Hpasserby/ZombieCrisis',
 'whiletest/zombie',
 'Franc1sco/sm-zombiereloaded-3-Franug-Edition',
 'redsunservers/SuperZombieFortress',
 'SPC-Some-Polish-Coders/PopHead',
 'pkpjpm/Zombie',
 'keymetrics/pm2-server-monit',
 'rico345100/unity-zombie-defense-fps-multiplayer-example',
 'HOKGroup/Zombie',
 'aa1000/GASTanksVsZombies',
 'grosser/zombie_passenger_killer',
 'eahs/ZombiePong',
 'HopsonCommunity/ZombieGame',
 'TheTurkeyDev/Call_Of_Minecraft-Zombies',
 'Blumlaut/RottenV',
 'rohangoel96/PlantsVsZombies-Game',
 'shishir99111/crypto-zombies',
 'dmcinnes/dead-valley',
 'sunziping2016/Qt-PlantsVsZombies',
 'GoogleCloudPlatform/appengine-scipy-zombie-apocalypse-python',
 'o3a/ZombieSmashersXNA4',
 'r3dxpl0it/ZombieBotV12',
 'CompleteUnityDeveloper2/6_Zombie_Runner',
 'rhelgeby/sm-zombiereloaded-3',
 'stoneharry/ZombiesProjectEmulator',
 'Doug-Pardee/LightZombie',
 'Zet0rz/nZombies-Unlimited',
 'rscaptainjack/ZOMBIES_CODE',
 'CSCI-E32/zombietranslator',
 'saurass/Zombie-DDoS',
 'travist/zombie-phantom',
 'raheem-cs/Zombie-Escape',
 'DeadlyApps/Unity-ECS-ZombieSimulator',
 'HelloWorld017/ZombieGame',
 'laughedelic/atom-ide-scala',
 'adinfinit/zombies-on-ice',
 'X-SLAYER/plants-vs-zombie-game-flutter',
 'Kenan2000/Secronom-Zombies',
 'hnspoc/Zombie',
 'jermay/crypto-kitties',
 'brunops/zombies-game',
 'SaroyaMan/Zombies-Stratego',
 'eduardtarassov/ZombieBirdGame',
 'PerfectScrash/Zombie-Plague-Special',
 'zahard/plants-vs-zombies',
 'colonelsalt/ZombieDeathBoomECS',
 'vector-wlc/AsmVsZombies',
 'samuelmaddock/zombie-escape',
 'ondras/trw',
 'dondido/zombie-breakout',
 'nZombies-Time/nZombies-Rezzurrection',
 'marcelosobral/Zombie_Apocalypse',
 'TheJosh/chaotic-rage',
 'samyk/skyjack',
 'fabiocaccamo/python-fsutil',
 'Bil369/YiQi-ZombieCompanyClassifier',
 'hartcode/zombie-hunter',
 'luis-rei97/-ZR-Zombie-Rank',
 'mileskin/zombie-jasmine-spike',
 'Shamsuzzaman321/zombiebotv14',
 'gdg-nova/game1',
 'NJU-TJL/PlantsVsZombies',
 'myENA/consul-zombie',
 'leandrovieiraa/FreeSurvivalZombieKit',
 'whitehatjr/zombie-crush-assets',
 'minusreality/virtual-reality',
 'IcyEngine/ZombieRP-shit',
 'JezuzLizard/Recompilable-gscs-for-BO2-zombies-and-multiplayer',
 '58115310/MinecraftVSZombies2Translations',
 'rico345100/unity-zombie-defense-fps-multiplayer-score-server',
 'magnars/parens-of-the-dead',
 'Jbleezy/BO2-Reimagined',
 'Zombiefied7/ZombiefiedZombieApocalpyse',
 'scalespeeder/DayZ-110-ZOMBIELAND-xml-mod-packs',
 'Nicksf13/zombieplague',
 'jianqizhao1992/ZombieShooter-Game',
 'TheGeogeo/CW_ZM_Trainer',
 'blazn420/BLAZN-CW-TOOL',
 'Rubery/RuberyKernel',
 'dtzxporter/Kronorium',
 'Corosauce/ZombieCraft'
]

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
