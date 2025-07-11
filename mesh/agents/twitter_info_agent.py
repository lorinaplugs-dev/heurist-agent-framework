import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class TwitterInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("APIDANCE_API_KEY")
        if not self.api_key:
            raise ValueError("APIDANCE_API_KEY environment variable is required")

        self.base_url = "https://api.apidance.pro"
        self.headers = {"apikey": self.api_key}

        self.metadata.update(
            {
                "name": "Twitter Profile Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent fetches a Twitter user's profile information and recent tweets. It's useful for getting project updates or tracking key opinion leaders (KOLs) in the space.",
                "external_apis": ["Twitter API"],
                "tags": ["Twitter"],
                "recommended": True,
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/Twitter.png",
                "examples": [
                    "Summarise recent updates of @heurist_ai",
                    "What has @elonmusk been tweeting lately?",
                    "Get the recent tweets from cz_binance",
                ],
                "credits": 2,
            }
        )

    def get_system_prompt(self) -> str:
        return """
        You are a specialized Twitter analyst that helps users get information about Twitter profiles and their recent tweets.

        Keep your analysis factual and concise. Only use the data provided. NEVER make up data that is not returned from the tool.
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_user_tweets",
                    "description": "Fetch recent tweets from a Twitter user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "Twitter username (with or without @) or numeric user ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of tweets to return",
                                "default": 10,
                            },
                        },
                        "required": ["username"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_twitter_detail",
                    "description": "Fetch detailed information about a specific tweet including replies and thread content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tweet_id": {
                                "type": "string",
                                "description": "The ID of the tweet to fetch details for",
                            },
                            "cursor": {
                                "type": "string",
                                "description": "Cursor for pagination through replies or threaded content",
                                "default": "",
                            },
                        },
                        "required": ["tweet_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_general_search",
                    "description": "Search for tweets using a query term",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "The search query term (e.g. eth)",
                            },
                            "cursor": {
                                "type": "string",
                                "description": "A pagination token to fetch the next page of results",
                                "default": "",
                            },
                        },
                        "required": ["q"],
                    },
                },
            },
        ]

    def get_twitter_user_endpoint(self) -> str:
        return f"{self.base_url}/1.1/users/show.json"

    def get_twitter_tweets_endpoint(self) -> str:
        return f"{self.base_url}/sapi/UserTweets"

    def get_twitter_detail_endpoint(self) -> str:
        return f"{self.base_url}/sapi/TweetDetail"

    def get_twitter_search_endpoint(self) -> str:
        return f"{self.base_url}/sapi/Search"

    # ------------------------------------------------------------------------
    #                       SHARED / UTILITY METHODS
    # ------------------------------------------------------------------------
    def _clean_username(self, username: str) -> str:
        """Remove @ symbol if present in username"""
        return username.strip().lstrip("@")

    def _is_numeric_id(self, input_str: str) -> bool:
        """Check if the input is a numeric ID"""
        return input_str.strip().isdigit()

    # ------------------------------------------------------------------------
    #                      TWITTER API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_user_id(self, identifier: str) -> Dict:
        """Fetch Twitter user ID and profile information using _api_request"""
        try:
            params = {}
            if self._is_numeric_id(identifier):
                params = {"user_id": identifier}
            else:
                clean_username = self._clean_username(identifier)
                params = {"screen_name": clean_username}

            logger.info(f"Fetching user profile for identifier: {identifier}")
            user_data = await self._api_request(
                url=self.get_twitter_user_endpoint(), method="GET", headers=self.headers, params=params
            )

            if "error" in user_data:
                logger.error(f"Error fetching user profile: {user_data['error']}")
                return user_data

            profile_info = {
                "id_str": user_data.get("id_str"),
                "name": user_data.get("name"),
                "screen_name": user_data.get("screen_name"),
                "description": user_data.get("description"),
                "followers_count": user_data.get("followers_count"),
                "friends_count": user_data.get("friends_count"),
                "statuses_count": user_data.get("statuses_count"),
                "profile_image_url": user_data.get("profile_image_url_https"),
                "verified": user_data.get("verified", False),
                "created_at": user_data.get("created_at"),
            }

            logger.info(f"Successfully fetched profile for user: {profile_info.get('screen_name')}")
            return {"profile": profile_info}

        except Exception as e:
            logger.error(f"Error in get_user_id: {e}")
            return {"error": f"Failed to fetch user profile: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_tweets(self, user_id: str, limit: int = 10) -> Dict:
        """Fetch recent tweets for a user by ID using _api_request"""
        try:
            params = {"user_id": user_id, "count": min(limit, 50)}

            logger.info(f"Fetching tweets for user_id: {user_id}, limit: {limit}")
            tweets_data = await self._api_request(
                url=self.get_twitter_tweets_endpoint(), method="GET", headers=self.headers, params=params
            )

            if "error" in tweets_data:
                logger.error(f"Error fetching tweets: {tweets_data['error']}")
                return tweets_data

            tweets = tweets_data.get("tweets", [])
            cleaned_tweets = []

            for tweet in tweets:
                cleaned_tweet = {
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", ""),
                    "engagement": {
                        "retweets": tweet.get("retweet_count", 0),
                        "likes": tweet.get("favorite_count", 0),
                        "replies": tweet.get("reply_count", 0) if "reply_count" in tweet else 0,
                    },
                }
                cleaned_tweets.append(cleaned_tweet)

            logger.info(f"Successfully fetched {len(cleaned_tweets)} tweets")
            return {"tweets": cleaned_tweets}

        except Exception as e:
            logger.error(f"Error in get_tweets: {e}")
            return {"error": f"Failed to fetch user tweets: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_tweet_detail(self, tweet_id: str, cursor: str = "") -> Dict:
        """Fetch detailed information about a specific tweet using _api_request"""
        try:
            params = {"tweet_id": tweet_id}
            if cursor:
                params["cursor"] = cursor

            logger.info(f"Fetching tweet details for tweet_id: {tweet_id}")
            tweet_data = await self._api_request(
                url=self.get_twitter_detail_endpoint(), method="GET", headers=self.headers, params=params
            )

            if "error" in tweet_data:
                logger.error(f"Error fetching tweet details: {tweet_data['error']}")
                return tweet_data

            result = {
                "pinned_tweet": tweet_data.get("pinned_tweet"),
                "tweets": tweet_data.get("tweets", []),
                "next_cursor": tweet_data.get("next_cursor_str"),
            }

            logger.info("Successfully fetched tweet details")
            return result

        except Exception as e:
            logger.error(f"Error in get_tweet_detail: {e}")
            return {"error": f"Failed to fetch tweet details: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def general_search(self, query: str, cursor: str = "") -> Dict:
        """Search for tweets using a query term using _api_request"""
        try:
            params = {"q": query}
            if cursor:
                params["cursor"] = cursor

            logger.info(f"Performing general search for query: {query}")
            search_data = await self._api_request(
                url=self.get_twitter_search_endpoint(), method="GET", headers=self.headers, params=params
            )

            if "error" in search_data:
                logger.error(f"Error in general search: {search_data['error']}")
                return search_data

            result = {
                "query": query,
                "tweets": search_data.get("tweets", []),
                "next_cursor": search_data.get("next_cursor_str"),
            }

            logger.info(f"Successfully completed search for query: {query}")
            return result

        except Exception as e:
            logger.error(f"Error in general_search: {e}")
            return {"error": f"Failed to search tweets: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle tool execution logic"""
        logger.info(f"Handling tool call: {tool_name} with args: {function_args}")

        if tool_name == "get_user_tweets":
            identifier = function_args.get("username")
            limit = min(function_args.get("limit", 10), 50)  # Cap at 50

            if not identifier:
                return {"error": "Missing 'username' parameter"}

            logger.info(f"Fetching tweets for identifier '{identifier}' with limit={limit}")
            self.push_update(
                {"identifier": identifier}, f"Looking up Twitter user: @{self._clean_username(identifier)}..."
            )

            # Get user profile first
            profile_result = await self.get_user_id(identifier)
            errors = self._handle_error(profile_result)
            if errors:
                return errors

            user_id = profile_result.get("profile", {}).get("id_str")
            if not user_id:
                return {"error": "Could not retrieve user ID"}

            # Get user tweets
            tweets_result = await self.get_tweets(user_id, limit)
            errors = self._handle_error(tweets_result)
            if errors:
                return errors

            return {
                "twitter_data": {
                    "profile": profile_result.get("profile", {}),
                    "tweets": tweets_result.get("tweets", []),
                }
            }

        elif tool_name == "get_twitter_detail":
            tweet_id = function_args.get("tweet_id")
            cursor = function_args.get("cursor", "")

            if not tweet_id:
                return {"error": "Missing 'tweet_id' parameter"}

            logger.info(f"Fetching tweet details for tweet_id '{tweet_id}'")

            tweet_detail_result = await self.get_tweet_detail(tweet_id, cursor)
            errors = self._handle_error(tweet_detail_result)
            if errors:
                return errors

            return {"tweet_data": tweet_detail_result}

        elif tool_name == "get_general_search":
            query = function_args.get("q")
            cursor = function_args.get("cursor", "")

            if not query:
                return {"error": "Missing 'q' parameter"}

            logger.info(f"Performing general search for query '{query}'")

            search_result = await self.general_search(query, cursor)
            errors = self._handle_error(search_result)
            if errors:
                return errors

            return {"search_data": search_result}

        else:
            error_msg = f"Unsupported tool: {tool_name}"
            logger.error(error_msg)
            return {"error": error_msg}
