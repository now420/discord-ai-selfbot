import discord
import time
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

class MyClient(discord.Client):
    def __init__(self):
        super().__init__()
        self.blenderbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.blenderbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.cooldowns = {}
        self.cooldown_time = 5

    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, message):
        if message.content.startswith('/ai'):
            user_id = message.author.id
            current_time = time.time()

            # Check if the user is on cooldown
            if user_id in self.cooldowns:
                time_since_last_command = current_time - self.cooldowns[user_id]
                if time_since_last_command < self.cooldown_time:
                    remaining_time = self.cooldown_time - time_since_last_command
                    await message.channel.send(f"You are on cooldown for {remaining_time:.1f} more seconds.")
                    return

            self.cooldowns[user_id] = current_time

            input_text = message.content[4:].strip()
            if not input_text:
                await message.channel.send("Please provide input after `/ai`.")
                return

            input_ids = self.blenderbot_tokenizer.encode("chat with blenderbot: " + input_text + " </s>", return_tensors="pt")
            output = self.blenderbot_model.generate(input_ids, max_length=1000, num_return_sequences=1, early_stopping=True)
            response = self.blenderbot_tokenizer.decode(output[0], skip_special_tokens=True)

            await message.reply(response)

client = MyClient()
client.run('your token here')
