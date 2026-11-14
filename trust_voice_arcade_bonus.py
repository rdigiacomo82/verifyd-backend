"""Pure deterministic educational game rounds for VeriFYD Circle Challenge Arcade.

The return's `correct` index is SERVER ONLY. API clients receive public_round().
Content age bands are instructional guidance, not age authorization.
"""
from __future__ import annotations
import hashlib
import random

BONUS_GAMES = {
    'math-orbit': {'title':'Math Orbit','subject':'Math','age':'6–9+','age_groups':['6-9','10-13','14-17','18+'],'unlock_points':0,'icon':'🪐','description':'Solve increasingly complex equations in a visual space lab.'},
    'eco-explorer': {'title':'Eco Explorer','subject':'Science','age':'8–12+','age_groups':['6-9','10-13','14-17','18+'],'unlock_points':0,'icon':'🌱','description':'Explore ecosystems, energy, habitats and science discoveries.'},
    'story-sleuth': {'title':'Story Sleuth','subject':'Reading','age':'9–adult','age_groups':['6-9','10-13','14-17','18+'],'unlock_points':100,'icon':'🔎','description':'Discover evidence and infer what happens next.'},
    'map-quest': {'title':'Map Quest','subject':'Geography','age':'10–adult','age_groups':['10-13','14-17','18+'],'unlock_points':160,'icon':'🗺️','description':'Navigate a graphic map using coordinates and directions.'},
    'code-runner': {'title':'Code Runner','subject':'Coding','age':'12–adult','age_groups':['10-13','14-17','18+'],'unlock_points':260,'icon':'🤖','description':'Plan sequences of robot instructions to reach a target.'},
    'budget-boss': {'title':'Budget Boss','subject':'Life Skills','age':'14–adult','age_groups':['14-17','18+'],'unlock_points':420,'icon':'🪙','description':'Practice budgets, exact change, discounts and responsible choices.'},
}
BONUS_IDS = tuple(BONUS_GAMES)
DIFFICULTY_MIN_POINTS = (0,120,350)


def required_points(game: str, level: int) -> int:
    if game not in BONUS_GAMES or type(level) is not int or level not in (0,1,2):
        raise ValueError('Invalid bonus game or level')
    return max(BONUS_GAMES[game]['unlock_points'], DIFFICULTY_MIN_POINTS[level])


def _rng(seed: str, game: str, level: int, round_index: int) -> random.Random:
    digest=hashlib.sha256(f'v2:{seed}:{game}:{level}:{round_index}'.encode('utf-8')).digest()
    return random.Random(int.from_bytes(digest[:16],'big'))


def _make_options(correct, distractors, r):
    values=[correct]
    for value in distractors:
        if value != correct and value not in values: values.append(value)
        if len(values) >= 4: break
    if len(values)<4: raise ValueError('Not enough unique choices')
    r.shuffle(values)
    return [{'id':str(i),'label':str(v)} for i,v in enumerate(values)],str(values.index(correct))

SCIENCE_QUESTIONS = [
    ('🌿','Plants', 'What do plants absorb from sunlight to grow?', 'Light energy', ['Magnetic energy','Sound waves','Moon dust']),
    ('💧','Water', 'What happens when water vapor cools into droplets?', 'Condensation', ['Melting','Erosion','Photosynthesis']),
    ('🐝','Pollination', 'How can a bee help a flowering plant?', 'Moves pollen between flowers', ['Creates soil from rock','Makes the sun brighter','Stops all rain']),
    ('🌍','Earth', 'Which layer of Earth contains most weather?', 'Atmosphere', ['Inner core','Mantle','Bedrock']),
    ('🦉','Food web', 'In a food web, what is the owl?', 'A consumer', ['A producer','A mineral','A source of sunlight']),
    ('💡','Energy', 'What does a solar panel convert into electricity?', 'Sunlight', ['Gravity','Air pressure','Sound']),
    ('🧲','Materials', 'Which material is attracted to a typical magnet?', 'Iron', ['Dry wood','Paper','Glass']),
    ('🌱','Ecosystem', 'What helps an ecosystem remain resilient?', 'Biodiversity', ['One species only','Removing all plants','Removing every insect']),
    ('☁️','Water cycle', 'What process moves water from rivers into the air?', 'Evaporation', ['Freezing','Deposition','Condensation']),
    ('🧊','Matter', 'What happens when ice absorbs enough heat?', 'It melts', ['It becomes iron','It disappears forever','It becomes sand']),
    ('🐢','Habitats', 'Why do wetlands support many species?', 'They offer varied habitats', ['They have no water','No plants grow there','They eliminate food chains']),
    ('🪵','Carbon', 'What do decomposers help return to the soil?', 'Nutrients', ['Plastic toys','Sunlight','Electricity']),
    ('🌋','Geology', 'What type of rock forms when magma cools?', 'Igneous rock', ['Sedimentary rock','Metamorphic rock','Fossilized wood']),
    ('🌡️','Climate', 'How is climate different from daily weather?', 'Climate is long-term patterns', ['Climate is one morning','Climate is only wind','Climate is a weather app']),
]
STORY_QUESTIONS = [
    ('📚','The library','Mina saw the library lights go off and the librarian turn the sign to CLOSED. She put her book in the return slot.','Why did Mina use the return slot?','The building had closed',['She forgot how to read','She was choosing a new book','The book was wet']),
    ('🌦️','The picnic','Ava packed sandwiches. Dark clouds gathered, and the wind picked up. She moved the picnic under the pavilion.','What did Ava expect?','Rain may arrive',['The sandwiches were too heavy','The pavilion was closed','The sun became hotter']),
    ('🚌','The bus','Devin arrived at the stop and saw people checking the time. The sign said the bus was delayed ten minutes.','Why were people checking the time?','They were waiting for a late bus',['They were counting cars','They lost their phones','They were buying food']),
    ('🎨','The mural','The artist chose blues for the river and warm oranges for the sunrise. Neighbors offered brushes and helped paint.','What can you infer about the project?','The community worked together',['Only one person helped','The artist refused help','The river is orange']),
    ('🌱','The garden','Jordan noticed the soil was dry and the leaves drooped. After watering the plant, the leaves looked healthier.','Why did Jordan water the plant?','It needed moisture',['The plant wanted less light','The pot was too large','The leaves were painted']),
    ('🔦','The trail','As sunset approached, Sam checked the map and put fresh batteries in the flashlight before hiking back.','Why did Sam check the flashlight?','The walk could become dark',['To make a phone call','Because the map was broken','Because the sun was rising']),
    ('🧩','The puzzle','The team tried one idea that failed. Instead of quitting, they compared clues and tested another strategy.','What skill did the team demonstrate?','Persistence',['Giving up','Ignoring evidence','Avoiding teamwork']),
    ('🚲','The repair','A bike tire was flat. Lee checked the tube and found a small puncture, then patched it.','What caused the flat tire?','A hole in the tube',['A missing bell','A bright headlight','A loose helmet']),
]
DIRECTIONS=[('North',0,-1),('South',0,1),('East',1,0),('West',-1,0)]
COMMANDS={'↑':(0,-1),'↓':(0,1),'→':(1,0),'←':(-1,0)}


def build_round(game: str, level: int, round_index: int, seed: str) -> dict:
    if game not in BONUS_GAMES or level not in (0,1,2) or not (0<=round_index<6):
        raise ValueError('Invalid round')
    r=_rng(seed,game,level,round_index)
    if game=='math-orbit':
        if level==0:
            a,b=r.randint(3,15),r.randint(2,12)
            op=r.choice(['+','−']); answer=a+b if op=='+' else a-b
        elif level==1:
            a,b=r.randint(3,12),r.randint(3,11)
            op='×'; answer=a*b
        else:
            b=r.randint(3,11);answer=r.randint(8,20);a=answer*b
            op='÷'
        opts,correct=_make_options(answer,[answer+1,answer-1,answer+ (3+level),answer-(3+level),answer+7],r)
        return {'prompt':f'Solve the mission: {a} {op} {b} = ?', 'options':opts,'correct':correct,
                'visual':{'type':'math','left':a,'right':b,'symbol':op,'icon':'🪐'},'hint':'Work through the operation, then select the result.'}
    if game=='eco-explorer':
        eligible=SCIENCE_QUESTIONS[:8] if level==0 else SCIENCE_QUESTIONS[2:12] if level==1 else SCIENCE_QUESTIONS[4:]
        index=(round_index+r.randint(0,len(eligible)-1)) % len(eligible)
        icon,topic,prompt,answer,distractors=eligible[index]
        opts,correct=_make_options(answer,distractors,r)
        return {'prompt':prompt,'options':opts,'correct':correct,'visual':{'type':'science','icon':icon,'topic':topic},'hint':'Think about cause and effect in nature.'}
    if game=='story-sleuth':
        icon,topic,story,prompt,answer,distractors=STORY_QUESTIONS[r.randrange(len(STORY_QUESTIONS))]
        opts,correct=_make_options(answer,distractors,r)
        return {'prompt':prompt,'options':opts,'correct':correct,'visual':{'type':'story','icon':icon,'topic':topic,'story':story},'hint':'Look for evidence in the short story.'}
    if game=='map-quest':
        size=4+level
        x,y=r.randint(1,size-2),r.randint(1,size-2)
        num_moves=1 if level==0 else 2 if level==1 else 3
        steps=[]
        for _ in range(num_moves):
            options=[d for d in DIRECTIONS if 0<=x+d[1]<size and 0<=y+d[2]<size]
            direction=r.choice(options)
            steps.append(direction[0]); x+=direction[1]; y+=direction[2]
        end=f'{chr(65+x)}{y+1}'
        wrong=[f'{chr(65+a)}{b+1}' for a in range(size) for b in range(size) if f'{chr(65+a)}{b+1}'!=end]
        r.shuffle(wrong)
        opts,correct=_make_options(end,wrong,r)
        start_x,start_y=x,y
        for direction in reversed(steps):
            dx,dy=next((d[1],d[2]) for d in DIRECTIONS if d[0]==direction)
            start_x-=dx;start_y-=dy
        return {'prompt':f'Start at {chr(65+start_x)}{start_y+1}. Go '+', then '.join(steps)+'. Where do you arrive?',
                'options':opts,'correct':correct,'visual':{'type':'map','size':size,'start':[start_x,start_y], 'steps':steps},
                'hint':'Use the north arrow; right is east and up is north.'}
    if game=='code-runner':
        size=4+level
        start=(r.randint(1,size-2),r.randint(1,size-2))
        length=2+level
        path=[];x,y=start
        for _ in range(length):
            possible=[(s,dx,dy) for s,(dx,dy) in COMMANDS.items() if 0<=x+dx<size and 0<=y+dy<size]
            step,dx,dy=r.choice(possible);x+=dx;y+=dy;path.append(step)
        target=(x,y)
        answer=' '.join(path)
        wrong=[]
        # Each distractor is unique and provably does NOT reach the destination.
        tries=0
        while len(wrong)<3 and tries<500:
            tries+=1
            candidate=[];cx,cy=start
            for _ in range(length):
                step,dx,dy=r.choice(list((s,*p) for s,p in COMMANDS.items()))
                candidate.append(step);cx+=dx;cy+=dy
            text=' '.join(candidate)
            if text!=answer and (cx,cy)!=target and text not in wrong:
                wrong.append(text)
        opts,correct=_make_options(answer,wrong,r)
        return {'prompt':f'Guide the robot from {chr(65+start[0])}{start[1]+1} to {chr(65+target[0])}{target[1]+1}. Which command sequence works?',
                'options':opts,'correct':correct,'visual':{'type':'robot','size':size,'start':start,'goal':target},
                'hint':'Follow each arrow in order. The target square must match exactly.'}
    if game=='budget-boss':
        if level==0:
            budget=10*r.randint(2,8);cost=budget-r.randint(4,18);op='change'
            prompt=f'You have ${budget}. You buy supplies for ${cost}. How much money remains?'
            answer=budget-cost
        elif level==1:
            first=r.randint(7,29);second=r.randint(5,24);budget=first+second+r.randint(4,21);cost=first+second
            op='basket';prompt=f'Your budget is ${budget}. A notebook is ${first} and a puzzle is ${second}. What is left?'
            answer=budget-cost
        else:
            price=5*r.randint(9,24);discount=price//5;cost=price-discount;budget=price+r.randint(8,30)
            op='discount';prompt=f'A ${price} item is 20% off. You have ${budget}. How much money remains after buying it?'
            answer=budget-cost
        opts,correct=_make_options(f'${answer}', [f'${answer+n}' for n in (-5,5,10,-10,1,-1)],r)
        return {'prompt':prompt,'options':opts,'correct':correct,'visual':{'type':'budget','budget':budget,'spent':cost,'mode':op},
                'hint':'Calculate the total price first; then subtract it from the available money.'}
    raise ValueError('Unsupported bonus game')


def public_round(game: str, level: int, round_index: int, seed: str) -> dict:
    data=build_round(game,level,round_index,seed)
    return {'round':round_index,'prompt':data['prompt'],'options':data['options'],'visual':data['visual'],'hint':data['hint']}


def verify_bonus_round(game: str, level: int, round_index: int, seed: str, proof) -> bool:
    return type(proof) is str and proof == build_round(game,level,round_index,seed)['correct']