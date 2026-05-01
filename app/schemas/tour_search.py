from datetime import date

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class TourSearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    data_min: date
    data_max: date
    num_adults: int = Field(ge=1, le=2)
    num_childs: int = Field(ge=0, le=2)
    birthdays: list[date] = Field(
        default_factory=list,
        validation_alias=AliasChoices("birthdays", "bithdays"),
    )
    budget: float = Field(gt=0)
    countries: list[str] = Field(min_length=1)
    num_nights: list[int] = Field(min_length=1)
    query: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_payload(self) -> "TourSearchRequest":
        if self.data_min > self.data_max:
            raise ValueError("data_min must be less than or equal to data_max")

        if len(self.birthdays) != self.num_childs:
            raise ValueError("birthdays count must match num_childs")

        if self.num_childs == 0 and self.birthdays:
            raise ValueError("birthdays must be empty when num_childs is 0")

        if any(day > self.data_min for day in self.birthdays):
            raise ValueError("child birthday cannot be greater than data_min")

        if not all(country.strip() for country in self.countries):
            raise ValueError("countries must contain non-empty values")

        if not all(night > 0 for night in self.num_nights):
            raise ValueError("num_nights must contain only positive values")

        if not self.query.strip():
            raise ValueError("query must not be blank")

        return self


class TourOffer(BaseModel):
    teztour_id: int
    price: float
    checkin_date: date
    nights: int


class TourSearchResponse(BaseModel):
    offers: list[TourOffer]
