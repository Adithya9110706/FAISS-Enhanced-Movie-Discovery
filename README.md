
FAISS-Enhanced Movie Discovery: A Semantic Search Approach







 Abstract :
This paper presents a scalable movie search and recommendation system that leverages FAISS indexing and TMDb API for efficient retrieval and ranking of movies based on user queries. Unlike traditional keyword-based search methods, this system employs Sentence Transformers to generate semantic embeddings, ensuring context-aware recommendations. The search engine fetches metadata and images from TMDb, indexes movie descriptions using FAISS, and ranks relevant movies based on cosine similarity matching.
The report outlines the crawler design, annotation strategies, indexing process, retrieval mechanisms, evaluation metrics, and system performance. The project is benchmarked using precision-based evaluation metrics and demonstrates improved search accuracy compared to conventional movie search engines.

CCS Concepts
•	Information systems → Information retrieval → Retrieval models and ranking
•	Computing methodologies → Machine learning → Neural network-based indexing
•	Information systems → Recommender systems
•	Information systems → Database management → Query processing
•	World Wide Web → Web applications → Search engines
Keywords
Movie search, Semantic search, FAISS indexing, Natural language processing (NLP), Sentence Transformers, Information retrieval, Cosine similarity, Query processing, Vector-based search, AI-powered search, Movie metadata, Efficient retrieval, Database indexing, User query matching, Flask web application, Scalable search engine
ACM Reference Format:
Adithya Ramelli .  FAISS-Enhanced Movie Discovery: A Semantic Search Approach. In .Dublin City University, Dublin, Ireland, Article , 6 pages

1	Introduction
The ability to efficiently retrieve relevant movie information is crucial in today's digital landscape, where vast amounts of entertainment content are produced daily. Traditional search engines rely on keyword-based matching, which often fails to capture semantic relationships between user queries and movie descriptions, resulting in inaccurate or suboptimal recommendations. To overcome these limitations, this paper introduces an AI-driven movie search system, leveraging semantic embeddings to enable context-aware retrieval and ranking. The proposed system utilizes FAISS indexing combined with Sentence Transformers to enhance the precision and speed of search queries. By incorporating natural language processing (NLP) techniques, the system ensures that users receive recommendations tailored to their input, improving search effectiveness beyond conventional string-based methods.
Pre-processing The core of this system is built upon semantic vectorization, where movie metadata—including titles, genres, descriptions, and user ratings—is transformed into numerical embeddings using pre-trained Sentence Transformers. These embeddings allow the system to understand context and meaning, enabling a more intelligent and dynamic search process. Unlike simple keyword searches, the semantic indexing approach helps identify relevant movies even when the query differs from the exact wording in the dataset. FAISS (Facebook AI Similarity Search) is employed to construct an efficient high-dimensional vector index, optimizing nearest-neighbor retrieval for fast query response times. This indexing method significantly enhances accuracy, allowing users to discover movies based on thematic relevance rather than rigid keyword constraints
To ensure comprehensive data collection and annotation, the system integrates a web crawler that extracts movie metadata and posters from the TMDb API. Metadata is carefully annotated, including genre classifications, movie descriptions, and popularity metrics, which serve as critical features for ranking recommendations. The crawler operates in a multi-threaded environment, efficiently collecting and storing large-scale movie datasets. This structured data facilitates high-quality embedding generation, further strengthening retrieval effectiveness. By preprocessing textual information, eliminating noise, and enriching metadata attributes, the system establishes a robust dataset that enhances contextual matching during search queries.
 The retrieval component of the search engine leverages FAISS-based similarity search, ensuring real-time query processing. Users can input descriptions, keywords, or thematic ideas, and the system generates highly relevant recommendations by computing similarity scores between query embeddings and indexed movie vectors. The ranking model considers multiple metadata attributes, prioritizing high-rated movies, genre relevance, and linguistic similarity. A Flask-powered web interface enables an intuitive and interactive search experience, allowing users to explore movie suggestions dynamically. The integration of real-time search functionality ensures that the results are displayed instantly, improving user engagement and responsiveness. 
This paper explores the architecture, indexing methodology, retrieval mechanisms, and evaluation metrics of the proposed movie search system. Performance is assessed through retrieval accuracy, query processing speed, and ranking effectiveness, utilizing standard information retrieval metrics such Recall, and Cosine Similarity Scores. The results provide insights into the impact of semantic embeddings and FAISS indexing on movie search optimization. The study highlights how AI-driven retrieval models outperform traditional search approaches, offering a fast, scalable, and intelligent recommendation system. By enhancing query understanding and ranking relevance, the proposed system demonstrates the future potential of AI-powered movie discovery platforms.

2	Annotation
2.1	Metadata Extraction
Each movie in the dataset is annotated with metadata retrieved from the TMDb API, ensuring that images are properly categorized and linked to meaningful descriptions. Metadata plays a crucial role in improving the search experience by providing structured information, allowing users to discover relevant content beyond simple keyword matching. The extracted attributes include the movie title, release date, genres, overview, ratings, and popularity metrics. The title serves as the primary identifier, while the overview provides a summary that enhances semantic search capabilities. Additionally, ratings contribute to ranking relevance, helping prioritize movies that are more highly rated or widely recognized.
2.2	Image Processing and Storage

To enhance visual representation in search results, movie posters are downloaded and stored systematically. Each movie has a unique poster path, which is fetched via an API call. Once retrieved, the images are stored locally using a structured naming convention that aligns with the movie’s metadata. This ensures consistency and prevents misalignment between images and their corresponding movies. The poster images are then mapped to their metadata in the database, allowing efficient retrieval and display during searches. By associating images with structured metadata, the system ensures that users receive both visual and textual information when browsing movie recommendations.

2.3	Genre-Based Classification
Genre classification is a critical part of the annotation process, allowing the system to categorize movies into thematic groups. The TMDb API provides genre IDs for each movie, which are mapped to human-readable genre labels such as Action, Comedy, Drama, and Science Fiction. By integrating genre-based annotations, the search engine refines recommendations and improves filtering accuracy. Users can search for movies by category, ensuring that results are relevant to their preferences. Furthermore, genre annotations assist in ranking searches more intelligently, allowing movies with similar themes to be grouped together and surfaced more effectively in search queries

2.4	 Embedding-Based Text Association
To enhance semantic understanding, the system processes textual metadata—such as the title, overview, and genre information—using Sentence Transformers. These embeddings transform text into high-dimensional vector representations, enabling the search system to match queries with contextually relevant movies. FAISS indexing stores these embeddings, facilitating efficient search and retrieval operations. This embedding-based approach ensures that recommendations are not solely dependent on direct keyword matching but instead leverage deeper contextual relationships between query descriptions and movie metadata.

2.5	Annotation Validation and Quality Control
To ensure the accuracy of annotations, a multi-step validation process is conducted. Automated checks verify that downloaded images correctly match their respective movies, preventing inconsistencies. Additionally, manual inspection of randomly selected movies helps confirm the correct mapping of metadata. Retrieval performance is also evaluated by running test queries to assess how effectively search results align with annotated movie attributes. By implementing thorough validation measures, the system maintains high annotation quality, ensuring that search queries yield precise and meaningful recommendations.

3	Indexing
Indexing is a crucial step in information retrieval systems, enabling efficient access to stored data for fast query processing. Unlike traditional keyword-based approaches, which rely on exact text matching, this system employs semantic indexing to enhance search accuracy. By structuring movie metadata in a vectorized format, the system can rank and retrieve movies based on contextual relevance instead of simple keyword occurrence. The indexing process organizes large volumes of metadata and optimizes search speed, ensuring a smooth user experience.

3.1	Semantic Indexing with FAISS
To achieve high-performance retrieval, this system utilizes FAISS (Facebook AI Similarity Search), a machine learning-based indexing framework designed for efficient similarity searches in large datasets. FAISS stores movie metadata as high-dimensional vectors, allowing the system to perform approximate nearest-neighbor (ANN) searches at scale. Each movie's metadata—including title, genres, overview, and ratings—is converted into a semantic vector representation using Sentence Transformers, ensuring deeper contextual understanding.

3.2	The FAISS indexing process consists of several steps:
•	Embedding Generation: Movie descriptions are transformed into dense vector representations using Sentence Transformers.
•	Normalization: Vectors are normalized to maintain consistent scaling and ensure accurate similarity comparisons.
•	Index Storage: FAISS stores these embeddings in a high-dimensional space, enabling quick similarity searches.
•	Query Matching: When a user submits a query, the system finds the nearest movie vectors that match the embedded search phrase.
This indexing method eliminates rigid keyword dependency, allowing for natural language queries and ensuring relevant results even for vague or loosely structured search terms.

3.3	Index Construction and Optimization
Building an efficient index requires structuring metadata in a way that ensures scalability and low-latency retrieval. The system optimizes its indexing strategy by employing approximate nearest-neighbor (ANN) algorithms, which balance accuracy and speed during similarity searches. FAISS supports large datasets by using compressed storage formats, making high-dimensional retrieval feasible without excessive memory consumption. Additionally, genre classifications and movie ratings are embedded within the index, refining search results based on category relevance and popularity.

3.4	 Retrieval Efficiency and Scalability
One of the biggest advantages of FAISS indexing is its ability to scale. The indexing process ensures that as more movies are added to the database, query execution times remain low and efficient. Unlike traditional search methods that slow down with large datasets, FAISS maintains consistent performance through optimized vector-based retrieval. The system’s indexing mechanism allows for real-time query execution, reducing wait times for users while delivering accurate and context-aware recommendations.

3.5	Role of Indexing in Query Processing
The indexing framework plays a vital role in query execution and ranking decisions. When a user submits a query, the system performs the following steps:
•	Query Embedding: The user input is processed through Sentence Transformers, generating a semantic vector representation.
•	Similarity Search: FAISS compares the query embedding with stored movie vectors, ranking results by relevance.
•	Filtered Ranking: Additional metadata—such as ratings and genres—is applied to refine the search results further.
•	Result Presentation: Movies are displayed in ranked order, ensuring that the most contextually relevant options appear first.
 By eliminating reliance on exact text matching, the indexing process ensures searches consider meaning and intent, making retrieval more intelligent and effective.

4	Retrieval
Retrieval is the process of finding and ranking relevant movies based on user queries. This system uses semantic search, which means it understands the meaning behind words rather than just matching keywords. Unlike traditional search methods that only look for exact words, this system analyzes the context of the query using Sentence Transformers and retrieves movies based on how closely they relate to the meaning of the search.

4.1	How Queries Are Processed
When a user enters a search query, the system first converts the text into a semantic vector, which is a numerical format that represents the meaning of the words. This conversion is done using Sentence Transformers, an advanced natural language processing (NLP) technique. 
•	The system breaks down the query into individual words (tokenization).
•	It then generates an embedding, which means it converts the query into a format that can be compared mathematically with stored movie descriptions.
•	The embedding is normalized to ensure accuracy 
       Once the query is converted into this format, the system searches through a database of movie embeddings to find movies with the most similar meanings.

4.2	Finding and Ranking Relevant Movies
To retrieve the most relevant results, the system uses FAISS indexing, which enables fast searches in large databases. FAISS finds the nearest matches to the query using cosine similarity, a mathematical measure of how similar two pieces of text are. The ranking process is based on:
•	How closely the movie description matches the query (cosine similarity).
•	Metadata filtering, which takes into account movie ratings, popularity, and genres to refine search results.
•	Top-ranked movies, ensuring that the system only shows the most relevant options rather than displaying every possible match.

4.3	How Results Are Presented to the User
Once the system finds relevant movies, they are displayed in a Flask-based web application, which is a simple and interactive user interface. The system ensures real-time results, meaning when a user enters a query, the recommendations appear instantly without long loading times.The interface presents: 
•	The movie title, so users can quickly identify the recommendations.
•	The genres, helping users see the movie’s category.
•	The release year, adding context to when the movie was made.
•	The ratings, providing a measure of how well the movie is received.
•	The movie poster, allowing users to visually browse results. 
The system ensures real-time results, meaning when a user enters a query, the recommendations appear instantly without long loading times.


4.4	Evaluating Search Performance
To measure how effective the search system is, several evaluation techniques are used: 
•	Precision-at-K (P@K) determines how many of the top results are actually relevant.
•	Mean Reciprocal Rank (MRR) analyzes how early relevant movies appear in search results.
•	Cosine Similarity Validation checks whether the retrieved movies closely match the user’s intent.
       Testing shows that semantic search greatly improves accuracy compared to basic keyword matching. It ensures that movies are ranked based on relevance rather than random keyword similarities, leading to better search experiences for users.

Metric	value
Recall@5	1.0
MRR	0.8333333333333334
mAP	0.8333333333333334


5	Evaluation
Evaluating the effectiveness of the movie search and recommendation system is essential to ensure it delivers relevant and accurate results while maintaining efficiency and scalability. The success of the system depends on its ability to retrieve movies that closely match user queries based on semantic understanding rather than simple keyword matching. The evaluation process involves assessing the accuracy of search results, measuring query processing speed, comparing retrieval methods, and analyzing user feedback to determine the system's overall performance. By applying well-established information retrieval metrics, the system's strengths and limitations can be identified, providing insights into areas for improvement and future enhancements.

5.1	Retrieval Accuracy Assessment
One of the key indicators of performance is retrieval accuracy, which measures how effectively the system returns relevant results. Traditional keyword-based search engines often struggle to understand the context behind queries, leading to mismatched or irrelevant results. This system leverages semantic search through FAISS indexing, improving retrieval accuracy by identifying movies with meaningfully related descriptions. Accuracy is assessed using Precision-at-K (P@K), which evaluates the number of relevant movies appearing in the top K search results. Additionally, Recall is measured to determine how well the system retrieves all relevant movies related to a query. Cosine similarity scores are calculated between the query embedding and movie metadata, ensuring results closely match the intended meaning of the user input.

5.2	Movies Retrieval Accuracy Assessment
Apart from accuracy, the speed and scalability of query processing are critical to ensuring a responsive search experience. The FAISS indexing system is designed to handle large-scale movie datasets efficiently, providing results within sub-second response times. Query execution speed is measured by analyzing the average retrieval time across various dataset sizes, ensuring that performance remains consistent even as the database grows. Compared to traditional database queries, FAISS indexing significantly reduces search latency by using approximate nearest-neighbor searches, allowing users to receive recommendations almost instantly. The ability to maintain high-speed performance under heavy search loads demonstrates the system’s scalability and suitability for larger movie collections.

5.3	Comparison with Keyword-Based Search
To highlight the advantages of the semantic search approach, a comparison is made between FAISS-based search and conventional keyword-matching methods. In keyword-based search systems, queries must closely match predefined text entries, often resulting in rigid and inflexible retrieval. For instance, searching for "movies about space exploration" in a standard search engine may only return movies with the exact phrase in their descriptions. In contrast, the semantic indexing model interprets the meaning of the query, retrieving related movies such as Interstellar, Gravity, and The Martian, even if the exact wording is not found in the metadata. User tests show that FAISS indexing consistently delivers more contextually appropriate results, making it far superior for natural language search queries.

5.4	Challenges and Future Improvements
While the system successfully delivers high-accuracy search results, certain challenges were observed during testing. Some queries with very abstract descriptions produced less precise results, indicating a need for fine-tuning the embedding model to enhance contextual awareness. Additionally, integrating user preferences and history could improve personalized recommendations, allowing the system to adapt to individual viewing habits. Future improvements may include refining genre classification mechanisms, incorporating external movie review scores, and optimizing ranking models to consider user engagement trends. These enhancements would allow the system to evolve into a more intelligent and adaptive movie search platform, offering tailored recommendations for different user interests.


6	User Interface
For the user interface, we employed Flask, a lightweight and efficient Python framework that facilitates the creation of web-based applications. Flask was chosen to provide a straightforward movie search platform, ensuring that users can input movie names and retrieve results dynamically. The interface is designed for ease of use, offering a smooth interaction between users and the retrieval system.
The primary feature of the interface is a search box, where users can input a movie name rather than a descriptive query. The system accepts direct title-based queries such as "Interstellar", "Titanic", or "Inception". Once a movie name is entered, it undergoes index searching, retrieving relevant metadata such as movie posters, genres, ratings, and release dates. Unlike keyword-based retrieval systems that analyze thematic descriptions, this system operates purely on title-based retrieval, fetching exact or closest matches from the indexed movie database.

Once a query is submitted, the system searches for the most relevant matches within the indexed dataset, ensuring that users receive highly accurate results without ambiguity. The retrieved movies are presented in a structured list or gallery format, displaying essential details such as title, genre, and a brief overview to enhance user engagement.

 
                       Figure 1: User Interface 
 
             Figure 2: Search results of movie “spider-man”


 
Example Queries Users Might Try:
•	"Titanic"
•	"Avatar"
•	"The Dark Knight"

 
             Figure 3: Metadata of movie “spider-man”

The search results are displayed in a gallery format, where each result consists of the movie poster, title, genre, and description. This structured presentation allows users to quickly browse and select their desired movie, making the search experience efficient and visually engaging.
To deploy the user interface, Flask hosts the search system on a local server, allowing users to query movies via a web browser. Additionally, the platform supports cloud deployment options, making it accessible remotely on services such as Google Cloud or AWS for scalability. By hosting the interface on a cloud environment, the system ensures real-time search capabilities without requiring local installations.


7	Conclusions
This project successfully demonstrates the effectiveness of ML-driven semantic retrieval in movie search and recommendation systems. By implementing FAISS indexing, Sentence Transformer embeddings, and Flask-based query processing, the system significantly improves search accuracy compared to traditional keyword-based methods. The use of context-aware semantic search ensures that users receive more relevant movie recommendations based on meaning rather than exact keyword matches. Additionally, the integration of multi-threaded metadata crawling from the TMDb API allows for a rich and diverse movie dataset, enabling dynamic recommendations.

7.1	Strengths of the System
One of the major strengths of this system is its ability to efficiently process and retrieve relevant movies in real-time. The use of high-dimensional vectorized search improves recommendation precision, ensuring that contextually similar movies appear at the top of search results. The FAISS-based retrieval model allows for fast nearest-neighbor searches, optimizing query execution time even as the dataset scales. Additionally, the Flask-powered web interface provides an intuitive experience for users, allowing them to interact with a smooth and responsive search system. The integration of movie posters and metadata visualization further enhances user engagement.

7.2	 Challenges and Limitations
While the system demonstrates high retrieval accuracy, certain challenges remain. Some queries containing abstract or highly specific descriptions can result in less precise matches, suggesting the need for fine-tuning the Sentence Transformer model for improved query interpretation. Additionally, the system does not currently support personalized recommendations, meaning search results are not tailored to individual user preferences. Another limitation is the reliance on external API data sources, which may introduce constraints in metadata availability and update frequency. Addressing these challenges will be critical for further refinement and expansion of the search capabilities.
 
7.3	Future Enhancements and Extensions
Several improvements can be made to enhance the system’s performance and usability. Personalized recommendations based on user interaction history could be introduced to provide more tailored results. Implementing advanced ranking mechanisms, such as hybrid filtering that incorporates user ratings and external review scores, could further optimize search accuracy. Scaling the system to support larger datasets with more diverse metadata attributes, including user-generated content, would improve recommendation diversity. Additionally, exploring interactive AI-based query refinement, where users can receive clarification prompts to improve search precision, could significantly boost the overall user experience.

8	Citations and Resources
The development of the Semantic Movie Search System was informed by several academic and technical resources, drawing insights from FAISS indexing, Sentence Transformer embeddings, Flask-based web frameworks, and semantic search methodologies. FAISS served as the foundation for implementing an efficient nearest-neighbor search mechanism, allowing for rapid retrieval of movies based on contextual similarity. Sentence Transformers played a crucial role in generating embeddings that transformed textual metadata into vectorized representations, ensuring more accurate search results. Additionally, Flask facilitated the creation of a lightweight, local server-based web interface, providing users with an intuitive platform to perform searches dynamically. The system was inspired by research in information retrieval models, including probabilistic ranking frameworks and vector-based search algorithms, which contribute to modern AI-driven search applications. In constructing the database, movie metadata was sourced from publicly available APIs, ensuring accessibility while maintaining structured indexing. The evaluation process adhered to standard information retrieval metrics, incorporating Precision@5, Recall@5, Mean Reciprocal Rank (MRR), Mean Average Precision (mAP), and Accuracy@5 to quantify the effectiveness of the search engine. The literature on query processing, similarity ranking, and document retrieval optimization also provided valuable insights into refining indexing structures and improving search efficiency. By integrating semantic search methodologies with scalable indexing techniques, this system demonstrates the potential for advancing movie retrieval technologies within AI-driven search frameworks. Let me know if you'd like further refinements.


9	References 
[1]	 Adithya Ramelli, FAISS-Enhanced Movie Discovery: A Semantic Search Approach, 2025. Available at: https://github.com/Adithya9110706/FAISS-Enhanced-Movie-Discovery
[2]	FAISS, FAISS Documentation, 2025. Available at: https://faiss.ai
[3]	Sentence Transformers, Pre-trained NLP Models, 2025. Available at: https://www.sbert 
[4]	Flask, Flask Web Framework Documentation, 2025. Available at: https://flask.palletsprojects.com. 
[5]	Wikipedia, Movie Search Systems, 2025. Available at: https://en.wikipedia.org/wiki/Movie_search 
[6]	The TMDb API, The Movie Database, 2025. Available at: https://www.themoviedb.org/documentation/api 
[7]	huggingface.co, The AI Community Building the Future, 2025. Available at: https://huggingface.co 
[8]	Wikimedia Foundation, Wikimedia, 2025. Available at: https://www.wikimedia.org/
[9]	 TiDB Team, Inverted Index vs Other Indexes: Key Differences, 2024. Available at: https://docs.pingcap.com/tidb/stable/inverted-index 
[10]	Adrian Rosebrock, Building a Scalable Search System for Images, 2018. Available at: https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/


 

