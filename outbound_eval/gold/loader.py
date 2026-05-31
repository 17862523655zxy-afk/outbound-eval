"""Gold conversation loader."""

import yaml
from pathlib import Path
from typing import Optional
from outbound_eval.gold.conversation import GoldConversation


class GoldConversationLoader:
    """Loader for gold conversations."""

    def __init__(self, library_dir: Optional[str] = None):
        """Initialize the loader.

        Args:
            library_dir: Directory containing gold conversation files.
        """
        if library_dir:
            self.library_dir = Path(library_dir)
        else:
            # /path/to/outbound_eval/outbound_eval/gold/loader.py
            # → /path/to/outbound_eval/data/gold/library
            package_dir = Path(__file__).parent.parent  # outbound_eval/
            self.library_dir = package_dir.parent / "data" / "gold" / "library"

    def load(self, conversation_id: str) -> GoldConversation:
        """Load a single conversation by ID.

        Args:
            conversation_id: Conversation ID (filename without .yaml)

        Returns:
            Loaded GoldConversation
        """
        # Search in all subdirectories
        for yaml_file in self.library_dir.rglob("*.yaml"):
            if yaml_file.stem == conversation_id:
                return self._load_file(yaml_file)

        raise FileNotFoundError(f"Conversation not found: {conversation_id}")

    def load_for_task(
        self, task_id: str, persona_type: Optional[str] = None
    ) -> list[GoldConversation]:
        """Load conversations for a specific task.

        Args:
            task_id: Task ID (maps to subdirectory name)
            persona_type: Optional filter by persona type

        Returns:
            List of GoldConversations
        """
        task_dir = self.library_dir / task_id
        if not task_dir.exists():
            return []

        conversations = []
        for conv_file in task_dir.glob("*.yaml"):
            try:
                conv = self._load_file(conv_file)
                if persona_type is None or conv.persona_type == persona_type:
                    conversations.append(conv)
            except Exception as e:
                print(f"Warning: Failed to load {conv_file}: {e}")

        return conversations

    def load_all(self) -> list[GoldConversation]:
        """Load all conversations.

        Returns:
            List of all GoldConversations
        """
        conversations = []

        for conv_file in self.library_dir.rglob("*.yaml"):
            try:
                conv = self._load_file(conv_file)
                conversations.append(conv)
            except Exception as e:
                print(f"Warning: Failed to load {conv_file}: {e}")

        return conversations

    def _load_file(self, filepath: Path) -> GoldConversation:
        """Load a conversation from file.

        Args:
            filepath: Path to YAML file

        Returns:
            GoldConversation object
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return GoldConversation(**data)

    def save(self, conversation: GoldConversation) -> None:
        """Save a conversation to file.

        Args:
            conversation: Conversation to save
        """
        task_dir = self.library_dir / conversation.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        filepath = task_dir / f"{conversation.conversation_id}.yaml"

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                conversation.model_dump(),
                f,
                allow_unicode=True,
                default_flow_style=False,
            )