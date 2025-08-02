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
                    "Search for 'bitcoin' (single word search)",
                    "Search for '#ETH' (hashtag search)",
                ],
                "credits": 2,
            }
        )

    def get_system_prompt(self) -> str:
        return """You are a specialized Twitter analyst that helps users get information about Twitter profiles and their recent tweets.
        
        IMPORTANT RULES:
        1. When using get_general_search, ONLY use single keywords, hashtags, or mentions. Multi-word searches will likely return empty results.
        2. Keep your analysis factual and concise. Only use the data provided.
        3. NEVER make up data that is not returned from the tool.
        4. If a search returns no results, suggest using a single keyword instead of multiple words.
        
        Search examples that work: 'bitcoin', '#ETH', '@username', '"exact phrase"'
        Search examples that fail: 'latest bitcoin news', 'what people think about ethereum'
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_user_tweets",
                    "description": "Fetch recent tweets from a specific Twitter user's timeline. This tool retrieves the most recent posts from a user's profile, including their own tweets and retweets. Use this when you want to see what a specific person or organization has been posting recently, track their updates, or analyze their Twitter activity patterns. The tool returns tweet content, engagement metrics (likes, retweets, replies), and timestamps. Maximum 50 tweets can be retrieved per request.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "Twitter username (with or without @) or numeric user ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of tweets to return (max: 50)",
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
                    "description": "Fetch detailed information about a specific tweet, including the full thread context and replies. This tool provides comprehensive data about a single tweet, including the original tweet content, any tweets in the same thread (if it's part of a conversation), and replies to the tweet. Use this when you need to understand the full context of a discussion, see how people are responding to a specific tweet, or analyze a Twitter thread. The tool returns the complete thread structure and engagement metrics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tweet_id": {
                                "type": "string",
                                "description": "The ID of the tweet to fetch details for",
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
                    "description": "Search for tweets using a SINGLE keyword, hashtag, or mention. WARNING: Multi-word searches often return EMPTY results on X/Twitter. ONLY use: single words (e.g., 'bitcoin'), hashtags (e.g., '#ETH'), mentions (e.g., '@username'), or exact phrases in quotes (e.g., '\"market crash\"'). NEVER use sentences or multiple unquoted words like 'latest bitcoin news' or 'what people think'. If you need to search for a complex topic, break it down into single keyword searches. This tool searches Twitter's public timeline for tweets matching your query. Each search query should be ONE concept only.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "The search query - MUST be a single keyword, hashtag (#example), mention (@username), or exact phrase in quotes. DO NOT use multiple words or sentences.",
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

    def _simplify_tweet_data(self, tweet: Dict) -> Dict:
        """Extract only essential tweet information"""
        simplified = {
            "id": tweet.get("id_str", ""),
            "text": tweet.get("text", ""),
            "created_at": tweet.get("created_at", ""),
            "author": {
                "id": tweet.get("user", {}).get("id_str", ""),
                "username": tweet.get("user", {}).get("screen_name", ""),
                "name": tweet.get("user", {}).get("name", ""),
            },
            "engagement": {
                "retweets": tweet.get("retweet_count", 0),
                "likes": tweet.get("favorite_count", 0),
                "replies": tweet.get("reply_count", 0),
            },
        }
        
        # Add thread/reply context if available
        if tweet.get("in_reply_to_status_id_str"):
            simplified["in_reply_to_tweet_id"] = tweet.get("in_reply_to_status_id_str")
            simplified["in_reply_to_user"] = tweet.get("in_reply_to_screen_name")
        
        # Add quoted tweet info if available
        if tweet.get("quoted_status"):
            simplified["quoted_tweet"] = {
                "id": tweet["quoted_status"].get("id_str", ""),
                "text": tweet["quoted_status"].get("text", ""),
                "author": tweet["quoted_status"].get("user", {}).get("screen_name", ""),
            }
            
        return simplified

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
            cleaned_tweets = [self._simplify_tweet_data(tweet) for tweet in tweets]

            logger.info(f"Successfully fetched {len(cleaned_tweets)} tweets")
            return {"tweets": cleaned_tweets}

        except Exception as e:
            logger.error(f"Error in get_tweets: {e}")
            return {"error": f"Failed to fetch user tweets: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_tweet_detail(self, tweet_id: str) -> Dict:
        """Fetch detailed information about a specific tweet using _api_request"""
        try:
            params = {"tweet_id": tweet_id}

            logger.info(f"Fetching tweet details for tweet_id: {tweet_id}")
            tweet_data = await self._api_request(
                url=self.get_twitter_detail_endpoint(), method="GET", headers=self.headers, params=params
            )

            if "error" in tweet_data:
                logger.error(f"Error fetching tweet details: {tweet_data['error']}")
                return tweet_data

            result = {
                "main_tweet": None,
                "thread_tweets": [],
                "replies": []
            }
            
            # Find the main tweet and organize thread/replies
            tweets = tweet_data.get("tweets", [])
            for tweet in tweets:
                simplified = self._simplify_tweet_data(tweet)
                if tweet.get("id_str") == tweet_id:
                    result["main_tweet"] = simplified
                elif tweet.get("in_reply_to_status_id_str") == tweet_id:
                    result["replies"].append(simplified)
                else:
                    result["thread_tweets"].append(simplified)

            logger.info("Successfully fetched tweet details")
            return result

        except Exception as e:
            logger.error(f"Error in get_tweet_detail: {e}")
            return {"error": f"Failed to fetch tweet details: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def general_search(self, query: str) -> Dict:
        """Search for tweets using a query term using _api_request"""
        try:
            # Warn if query appears to be multi-word without quotes
            if ' ' in query and not (query.startswith('"') and query.endswith('"')):
                logger.warning(f"Multi-word search query detected: '{query}'. This may return empty results.")
            
            params = {"q": query}

            logger.info(f"Performing general search for query: {query}")
            search_data = await self._api_request(
                url=self.get_twitter_search_endpoint(), method="GET", headers=self.headers, params=params
            )

            if "error" in search_data:
                logger.error(f"Error in general search: {search_data['error']}")
                return search_data

            tweets = search_data.get("tweets", [])
            simplified_tweets = [self._simplify_tweet_data(tweet) for tweet in tweets]

            result = {
                "query": query,
                "tweets": simplified_tweets,
                "result_count": len(simplified_tweets)
            }
            
            # Add warning if no results found
            if len(simplified_tweets) == 0:
                result["warning"] = "No results found. If you used multiple words, try a single keyword, hashtag (#example), or mention (@username) instead."

            logger.info(f"Successfully completed search for query: {query}, found {len(simplified_tweets)} results")
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

            if not tweet_id:
                return {"error": "Missing 'tweet_id' parameter"}

            logger.info(f"Fetching tweet details for tweet_id '{tweet_id}'")

            tweet_detail_result = await self.get_tweet_detail(tweet_id)
            errors = self._handle_error(tweet_detail_result)
            if errors:
                return errors

            return {"tweet_data": tweet_detail_result}

        elif tool_name == "get_general_search":
            query = function_args.get("q")

            if not query:
                return {"error": "Missing 'q' parameter"}

            # Log warning for multi-word queries
            if ' ' in query and not (query.startswith('"') and query.endswith('"')):
                logger.warning(f"Multi-word search query: '{query}'. Suggesting single keyword search.")
                self.push_update(
                    {"query": query}, 
                    "Warning: Multi-word searches often return empty results. Searching anyway..."
                )

            logger.info(f"Performing general search for query '{query}'")

            search_result = await self.general_search(query)
            errors = self._handle_error(search_result)
            if errors:
                return errors

            return {"search_data": search_result}

        else:
            error_msg = f"Unsupported tool: {tool_name}"
            logger.error(error_msg)
            return {"error": error_msg}
